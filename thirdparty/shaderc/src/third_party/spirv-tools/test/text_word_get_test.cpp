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

#define TAB "\t"
#define NEWLINE "\n"
#define BACKSLASH R"(\)"
#define QUOTE R"(")"

TEST(TextWordGet, NullTerminator) {
  std::string word;
  spv_position_t endPosition = {};
  ASSERT_EQ(
      SPV_SUCCESS,
      AssemblyContext(AutoText("Word"), nullptr).getWord(&word, &endPosition));
  ASSERT_EQ(4u, endPosition.column);
  ASSERT_EQ(0u, endPosition.line);
  ASSERT_EQ(4u, endPosition.index);
  ASSERT_STREQ("Word", word.c_str());
}

TEST(TextWordGet, TabTerminator) {
  std::string word;
  spv_position_t endPosition = {};
  ASSERT_EQ(SPV_SUCCESS, AssemblyContext(AutoText("Word\t"), nullptr)
                             .getWord(&word, &endPosition));
  ASSERT_EQ(4u, endPosition.column);
  ASSERT_EQ(0u, endPosition.line);
  ASSERT_EQ(4u, endPosition.index);
  ASSERT_STREQ("Word", word.c_str());
}

TEST(TextWordGet, SpaceTerminator) {
  std::string word;
  spv_position_t endPosition = {};
  ASSERT_EQ(
      SPV_SUCCESS,
      AssemblyContext(AutoText("Word "), nullptr).getWord(&word, &endPosition));
  ASSERT_EQ(4u, endPosition.column);
  ASSERT_EQ(0u, endPosition.line);
  ASSERT_EQ(4u, endPosition.index);
  ASSERT_STREQ("Word", word.c_str());
}

TEST(TextWordGet, SemicolonTerminator) {
  std::string word;
  spv_position_t endPosition = {};
  ASSERT_EQ(
      SPV_SUCCESS,
      AssemblyContext(AutoText("Wo;rd"), nullptr).getWord(&word, &endPosition));
  ASSERT_EQ(2u, endPosition.column);
  ASSERT_EQ(0u, endPosition.line);
  ASSERT_EQ(2u, endPosition.index);
  ASSERT_STREQ("Wo", word.c_str());
}

TEST(TextWordGet, NoTerminator) {
  const std::string full_text = "abcdefghijklmn";
  for (size_t len = 1; len <= full_text.size(); ++len) {
    std::string word;
    spv_text_t text = {full_text.data(), len};
    spv_position_t endPosition = {};
    ASSERT_EQ(SPV_SUCCESS,
              AssemblyContext(&text, nullptr).getWord(&word, &endPosition));
    ASSERT_EQ(0u, endPosition.line);
    ASSERT_EQ(len, endPosition.column);
    ASSERT_EQ(len, endPosition.index);
    ASSERT_EQ(full_text.substr(0, len), word);
  }
}

TEST(TextWordGet, MultipleWords) {
  AutoText input("Words in a sentence");
  AssemblyContext data(input, nullptr);

  spv_position_t endPosition = {};
  const char* words[] = {"Words", "in", "a", "sentence"};

  std::string word;
  for (uint32_t wordIndex = 0; wordIndex < 4; ++wordIndex) {
    ASSERT_EQ(SPV_SUCCESS, data.getWord(&word, &endPosition));
    ASSERT_EQ(strlen(words[wordIndex]),
              endPosition.column - data.position().column);
    ASSERT_EQ(0u, endPosition.line);
    ASSERT_EQ(strlen(words[wordIndex]),
              endPosition.index - data.position().index);
    ASSERT_STREQ(words[wordIndex], word.c_str());

    data.setPosition(endPosition);
    if (3 != wordIndex) {
      ASSERT_EQ(SPV_SUCCESS, data.advance());
    } else {
      ASSERT_EQ(SPV_END_OF_STREAM, data.advance());
    }
  }
}

TEST(TextWordGet, QuotesAreKept) {
  AutoText input(R"("quotes" "around words")");
  const char* expected[] = {R"("quotes")", R"("around words")"};
  AssemblyContext data(input, nullptr);

  std::string word;
  spv_position_t endPosition = {};
  ASSERT_EQ(SPV_SUCCESS, data.getWord(&word, &endPosition));
  EXPECT_EQ(8u, endPosition.column);
  EXPECT_EQ(0u, endPosition.line);
  EXPECT_EQ(8u, endPosition.index);
  EXPECT_STREQ(expected[0], word.c_str());

  // Move to the next word.
  data.setPosition(endPosition);
  data.seekForward(1);

  ASSERT_EQ(SPV_SUCCESS, data.getWord(&word, &endPosition));
  EXPECT_EQ(23u, endPosition.column);
  EXPECT_EQ(0u, endPosition.line);
  EXPECT_EQ(23u, endPosition.index);
  EXPECT_STREQ(expected[1], word.c_str());
}

TEST(TextWordGet, QuotesBetweenWordsActLikeGlue) {
  AutoText input(R"(quotes" "between words)");
  const char* expected[] = {R"(quotes" "between)", "words"};
  AssemblyContext data(input, nullptr);

  std::string word;
  spv_position_t endPosition = {};
  ASSERT_EQ(SPV_SUCCESS, data.getWord(&word, &endPosition));
  EXPECT_EQ(16u, endPosition.column);
  EXPECT_EQ(0u, endPosition.line);
  EXPECT_EQ(16u, endPosition.index);
  EXPECT_STREQ(expected[0], word.c_str());

  // Move to the next word.
  data.setPosition(endPosition);
  data.seekForward(1);

  ASSERT_EQ(SPV_SUCCESS, data.getWord(&word, &endPosition));
  EXPECT_EQ(22u, endPosition.column);
  EXPECT_EQ(0u, endPosition.line);
  EXPECT_EQ(22u, endPosition.index);
  EXPECT_STREQ(expected[1], word.c_str());
}

TEST(TextWordGet, QuotingWhitespace) {
  AutoText input(QUOTE "white " NEWLINE TAB " space" QUOTE);
  // Whitespace surrounded by quotes acts like glue.
  std::string word;
  spv_position_t endPosition = {};
  ASSERT_EQ(SPV_SUCCESS,
            AssemblyContext(input, nullptr).getWord(&word, &endPosition));
  EXPECT_EQ(input.str.length(), endPosition.column);
  EXPECT_EQ(0u, endPosition.line);
  EXPECT_EQ(input.str.length(), endPosition.index);
  EXPECT_EQ(input.str, word);
}

TEST(TextWordGet, QuoteAlone) {
  AutoText input(QUOTE);
  std::string word;
  spv_position_t endPosition = {};
  ASSERT_EQ(SPV_SUCCESS,
            AssemblyContext(input, nullptr).getWord(&word, &endPosition));
  ASSERT_EQ(1u, endPosition.column);
  ASSERT_EQ(0u, endPosition.line);
  ASSERT_EQ(1u, endPosition.index);
  ASSERT_STREQ(QUOTE, word.c_str());
}

TEST(TextWordGet, EscapeAlone) {
  AutoText input(BACKSLASH);
  std::string word;
  spv_position_t endPosition = {};
  ASSERT_EQ(SPV_SUCCESS,
            AssemblyContext(input, nullptr).getWord(&word, &endPosition));
  ASSERT_EQ(1u, endPosition.column);
  ASSERT_EQ(0u, endPosition.line);
  ASSERT_EQ(1u, endPosition.index);
  ASSERT_STREQ(BACKSLASH, word.c_str());
}

TEST(TextWordGet, EscapeAtEndOfInput) {
  AutoText input("word" BACKSLASH);
  std::string word;
  spv_position_t endPosition = {};
  ASSERT_EQ(SPV_SUCCESS,
            AssemblyContext(input, nullptr).getWord(&word, &endPosition));
  ASSERT_EQ(5u, endPosition.column);
  ASSERT_EQ(0u, endPosition.line);
  ASSERT_EQ(5u, endPosition.index);
  ASSERT_STREQ("word" BACKSLASH, word.c_str());
}

TEST(TextWordGet, Escaping) {
  AutoText input("w" BACKSLASH QUOTE "o" BACKSLASH NEWLINE "r" BACKSLASH ";d");
  std::string word;
  spv_position_t endPosition = {};
  ASSERT_EQ(SPV_SUCCESS,
            AssemblyContext(input, nullptr).getWord(&word, &endPosition));
  ASSERT_EQ(10u, endPosition.column);
  ASSERT_EQ(0u, endPosition.line);
  ASSERT_EQ(10u, endPosition.index);
  ASSERT_EQ(input.str, word);
}

TEST(TextWordGet, EscapingEscape) {
  AutoText input("word" BACKSLASH BACKSLASH " abc");
  std::string word;
  spv_position_t endPosition = {};
  ASSERT_EQ(SPV_SUCCESS,
            AssemblyContext(input, nullptr).getWord(&word, &endPosition));
  ASSERT_EQ(6u, endPosition.column);
  ASSERT_EQ(0u, endPosition.line);
  ASSERT_EQ(6u, endPosition.index);
  ASSERT_STREQ("word" BACKSLASH BACKSLASH, word.c_str());
}

TEST(TextWordGet, CRLF) {
  AutoText input("abc\r\nd");
  AssemblyContext data(input, nullptr);
  std::string word;
  spv_position_t pos = {};
  ASSERT_EQ(SPV_SUCCESS, data.getWord(&word, &pos));
  EXPECT_EQ(3u, pos.column);
  EXPECT_STREQ("abc", word.c_str());
  data.setPosition(pos);
  data.advance();
  ASSERT_EQ(SPV_SUCCESS, data.getWord(&word, &pos));
  EXPECT_EQ(1u, pos.column);
  EXPECT_STREQ("d", word.c_str());
}

}  // namespace
}  // namespace spvtools
