// Copyright (c) 2015-2016 The Khronos Group Inc.
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

#include <string>

#include "test/unit_spirv.h"

namespace spvtools {
namespace {

using spvtest::AutoText;

TEST(TextAdvance, LeadingNewLines) {
  AutoText input("\n\nWord");
  AssemblyContext data(input, nullptr);
  ASSERT_EQ(SPV_SUCCESS, data.advance());
  ASSERT_EQ(0u, data.position().column);
  ASSERT_EQ(2u, data.position().line);
  ASSERT_EQ(2u, data.position().index);
}

TEST(TextAdvance, LeadingSpaces) {
  AutoText input("    Word");
  AssemblyContext data(input, nullptr);
  ASSERT_EQ(SPV_SUCCESS, data.advance());
  ASSERT_EQ(4u, data.position().column);
  ASSERT_EQ(0u, data.position().line);
  ASSERT_EQ(4u, data.position().index);
}

TEST(TextAdvance, LeadingTabs) {
  AutoText input("\t\t\tWord");
  AssemblyContext data(input, nullptr);
  ASSERT_EQ(SPV_SUCCESS, data.advance());
  ASSERT_EQ(3u, data.position().column);
  ASSERT_EQ(0u, data.position().line);
  ASSERT_EQ(3u, data.position().index);
}

TEST(TextAdvance, LeadingNewLinesSpacesAndTabs) {
  AutoText input("\n\n\t  Word");
  AssemblyContext data(input, nullptr);
  ASSERT_EQ(SPV_SUCCESS, data.advance());
  ASSERT_EQ(3u, data.position().column);
  ASSERT_EQ(2u, data.position().line);
  ASSERT_EQ(5u, data.position().index);
}

TEST(TextAdvance, LeadingWhitespaceAfterCommentLine) {
  AutoText input("; comment\n \t \tWord");
  AssemblyContext data(input, nullptr);
  ASSERT_EQ(SPV_SUCCESS, data.advance());
  ASSERT_EQ(4u, data.position().column);
  ASSERT_EQ(1u, data.position().line);
  ASSERT_EQ(14u, data.position().index);
}

TEST(TextAdvance, EOFAfterCommentLine) {
  AutoText input("; comment");
  AssemblyContext data(input, nullptr);
  ASSERT_EQ(SPV_END_OF_STREAM, data.advance());
}

TEST(TextAdvance, NullTerminator) {
  AutoText input("");
  AssemblyContext data(input, nullptr);
  ASSERT_EQ(SPV_END_OF_STREAM, data.advance());
}

TEST(TextAdvance, NoNullTerminatorAfterCommentLine) {
  std::string input = "; comment|padding beyond the end";
  spv_text_t text = {input.data(), 9};
  AssemblyContext data(&text, nullptr);
  ASSERT_EQ(SPV_END_OF_STREAM, data.advance());
  EXPECT_EQ(9u, data.position().index);
}

TEST(TextAdvance, NoNullTerminator) {
  spv_text_t text = {"OpNop\nSomething else in memory", 6};
  AssemblyContext data(&text, nullptr);
  const spv_position_t line_break = {1u, 5u, 5u};
  data.setPosition(line_break);
  ASSERT_EQ(SPV_END_OF_STREAM, data.advance());
}

// Invokes AssemblyContext::advance() on text, asserts success, and returns
// AssemblyContext::position().
spv_position_t PositionAfterAdvance(const char* text) {
  AutoText input(text);
  AssemblyContext data(input, nullptr);
  EXPECT_EQ(SPV_SUCCESS, data.advance());
  return data.position();
}

TEST(TextAdvance, SkipOverCR) {
  const auto pos = PositionAfterAdvance("\rWord");
  EXPECT_EQ(1u, pos.column);
  EXPECT_EQ(0u, pos.line);
  EXPECT_EQ(1u, pos.index);
}

TEST(TextAdvance, SkipOverCRs) {
  const auto pos = PositionAfterAdvance("\r\r\rWord");
  EXPECT_EQ(3u, pos.column);
  EXPECT_EQ(0u, pos.line);
  EXPECT_EQ(3u, pos.index);
}

TEST(TextAdvance, SkipOverCRLF) {
  const auto pos = PositionAfterAdvance("\r\nWord");
  EXPECT_EQ(0u, pos.column);
  EXPECT_EQ(1u, pos.line);
  EXPECT_EQ(2u, pos.index);
}

TEST(TextAdvance, SkipOverCRLFs) {
  const auto pos = PositionAfterAdvance("\r\n\r\nWord");
  EXPECT_EQ(0u, pos.column);
  EXPECT_EQ(2u, pos.line);
  EXPECT_EQ(4u, pos.index);
}
}  // namespace
}  // namespace spvtools
