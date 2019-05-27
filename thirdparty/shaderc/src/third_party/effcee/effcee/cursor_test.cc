// Copyright 2017 The Effcee Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "gmock/gmock.h"

#include "cursor.h"

namespace {

using effcee::Cursor;
using effcee::LineMessage;
using effcee::StringPiece;
using ::testing::Eq;
using ::testing::HasSubstr;

// text method

// remaining and Advance methods
TEST(Cursor, AdvanceReturnsTheCursorItself) {
  Cursor c("foo");
  EXPECT_THAT(&c.Advance(1), Eq(&c));
}

TEST(Cursor, RemainingBeginsEqualToText) {
  const char* original = "The Smiths";
  Cursor c(original);
  EXPECT_THAT(c.remaining().begin(), Eq(original));
}

TEST(Cursor, RemainingDiminishesByPreviousAdvanceCalls) {
  const char* original = "The Smiths are a great 80s band";
  Cursor c(original);
  c.Advance(4);
  EXPECT_THAT(c.remaining(), Eq("Smiths are a great 80s band"));
  EXPECT_THAT(c.remaining().begin(), Eq(original + 4));
  c.Advance(11);
  EXPECT_THAT(c.remaining(), Eq("a great 80s band"));
  EXPECT_THAT(c.remaining().begin(), Eq(original + 15));
  c.Advance(c.remaining().size());
  EXPECT_THAT(c.remaining(), Eq(""));
  EXPECT_THAT(c.remaining().begin(), Eq(original + 31));
}

// Exhausted method

TEST(Cursor, ExhaustedImmediatelyWhenStartingWithEmptyString) {
  Cursor c("");
  EXPECT_TRUE(c.Exhausted());
}

TEST(Cursor, ExhaustedWhenRemainingIsEmpty) {
  Cursor c("boo");
  EXPECT_FALSE(c.Exhausted());
  c.Advance(2);
  EXPECT_FALSE(c.Exhausted());
  c.Advance(1);
  EXPECT_TRUE(c.Exhausted());
}

// RestOfLine method

TEST(Cursor, RestOfLineOnEmptyReturnsEmpty) {
  const char* original = "";
  Cursor c(original);
  EXPECT_THAT(c.RestOfLine(), Eq(""));
  EXPECT_THAT(c.RestOfLine().begin(), Eq(original));
}

TEST(Cursor, RestOfLineWithoutNewline) {
  Cursor c("The end");
  EXPECT_THAT(c.RestOfLine(), Eq("The end"));
}

TEST(Cursor, RestOfLineGetsLineUpToAndIncludingNewline) {
  Cursor c("The end\nOf an era");
  EXPECT_THAT(c.RestOfLine(), Eq("The end\n"));
}

TEST(Cursor, RestOfLineGetsOnlyFromRemainingText) {
  Cursor c("The end\nOf an era");
  c.Advance(4);
  EXPECT_THAT(c.remaining(), Eq("end\nOf an era"));
  EXPECT_THAT(c.RestOfLine(), Eq("end\n"));
}

// AdvanceLine and line_num methods

TEST(Cursor, AdvanceLineReturnsTheCursorItself) {
  Cursor c("foo\nbar");
  EXPECT_THAT(&c.AdvanceLine(), Eq(&c));
}

TEST(Cursor, AdvanceLineWalksThroughTextByLineAndCountsLines) {
  const char* original = "The end\nOf an era\nIs here";
  Cursor c(original);
  EXPECT_THAT(c.line_num(), Eq(1));
  c.AdvanceLine();
  EXPECT_THAT(c.line_num(), Eq(2));
  EXPECT_THAT(c.remaining(), Eq("Of an era\nIs here"));
  EXPECT_THAT(c.remaining().begin(), Eq(original + 8));
  c.AdvanceLine();
  EXPECT_THAT(c.line_num(), Eq(3));
  EXPECT_THAT(c.remaining(), Eq("Is here"));
  EXPECT_THAT(c.remaining().begin(), Eq(original + 18));
  c.AdvanceLine();
  EXPECT_THAT(c.line_num(), Eq(4));
  EXPECT_THAT(c.remaining(), Eq(""));
  EXPECT_THAT(c.remaining().begin(), Eq(original + 25));
}

TEST(Cursor, AdvanceLineIsNoopAfterEndIsReached) {
  Cursor c("One\nTwo");
  c.AdvanceLine();
  EXPECT_THAT(c.line_num(), Eq(2));
  EXPECT_THAT(c.remaining(), Eq("Two"));
  c.AdvanceLine();
  EXPECT_THAT(c.line_num(), Eq(3));
  EXPECT_THAT(c.remaining(), Eq(""));
  c.AdvanceLine();
  EXPECT_THAT(c.line_num(), Eq(3));
  EXPECT_THAT(c.remaining(), Eq(""));
}

// LineMessage free function.

TEST(LineMessage, SubtextIsFirst) {
  StringPiece text("Foo\nBar");
  StringPiece subtext(text.begin(), 3);
  EXPECT_THAT(LineMessage(text, subtext, "loves quiche"),
              Eq(":1:1: loves quiche\nFoo\n^\n"));
}

TEST(LineMessage, SubtextDoesNotEndInNewline) {
  StringPiece text("Foo\nBar");
  StringPiece subtext(text.begin()+4, 3);
  EXPECT_THAT(LineMessage(text, subtext, "loves quiche"),
              Eq(":2:1: loves quiche\nBar\n^\n"));
}

TEST(LineMessage, SubtextPartwayThroughItsLine) {
  StringPiece text("Food Life\nBar");
  StringPiece subtext(text.begin() + 5, 3); // "Lif"
  EXPECT_THAT(LineMessage(text, subtext, "loves quiche"),
              Eq(":1:6: loves quiche\nFood Life\n     ^\n"));
}

TEST(LineMessage, SubtextOnSubsequentLine) {
  StringPiece text("Food Life\nBar Fight\n");
  StringPiece subtext(text.begin() + 14, 5); // "Fight"
  EXPECT_THAT(LineMessage(text, subtext, "loves quiche"),
              Eq(":2:5: loves quiche\nBar Fight\n    ^\n"));
}

TEST(LineMessage, SubtextIsEmptyAndInMiddle) {
  StringPiece text("Food");
  StringPiece subtext(text.begin() + 2, 0);
  EXPECT_THAT(LineMessage(text, subtext, "loves quiche"),
              Eq(":1:3: loves quiche\nFood\n  ^\n"));
}

TEST(LineMessage, SubtextIsEmptyAndAtVeryEnd) {
  StringPiece text("Food");
  StringPiece subtext(text.begin() + 4, 0);
  EXPECT_THAT(LineMessage(text, subtext, "loves quiche"),
              Eq(":1:5: loves quiche\nFood\n    ^\n"));
}

}  // namespace
