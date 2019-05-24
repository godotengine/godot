// Copyright 2015 The Shaderc Authors. All rights reserved.
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

#include "libshaderc_util/string_piece.h"

#include <gtest/gtest.h>
#include <sstream>
#include <unordered_map>

#include "death_test.h"

namespace {

using shaderc_util::string_piece;

TEST(string_piece, creation) {
  std::string my_string("std::string");
  const char* my_c_string = "c::string";

  string_piece my_string_piece(my_string);
  string_piece my_c_string_piece(my_c_string);
  string_piece my_partial_c_string_piece(my_c_string, my_c_string + 3);
  string_piece my_string_piece_string_piece(my_string_piece);

  EXPECT_EQ("std::string", my_string_piece);
  EXPECT_EQ("c::string", my_c_string_piece);
  EXPECT_EQ("c::", my_partial_c_string_piece);
  EXPECT_EQ("std::string", my_string_piece_string_piece);
}

TEST(string_piece, creation_with_empty_data) {
  string_piece my_string_piece(nullptr, nullptr);
  EXPECT_EQ("", my_string_piece);
}

TEST(string_piece, creation_with_nullptr) {
  string_piece my_string_piece(nullptr);
  EXPECT_EQ("", my_string_piece);
}

TEST(string_pieceDeathTest, creation_causing_assert) {
  EXPECT_DEBUG_DEATH_IF_SUPPORTED(string_piece("my_cstring", nullptr), ".*");
  EXPECT_DEBUG_DEATH_IF_SUPPORTED(string_piece(nullptr, "my_cstring"), ".*");
}

TEST(string_pieceDeathTest, front) {
  EXPECT_DEBUG_DEATH_IF_SUPPORTED(string_piece(nullptr).front(), "Assertion");
  EXPECT_DEBUG_DEATH_IF_SUPPORTED(string_piece(nullptr, nullptr).front(),
                                  "Assertion");
  EXPECT_DEBUG_DEATH_IF_SUPPORTED(string_piece("").front(), "Assertion");
  string_piece s("nonempty");
  s.clear();
  EXPECT_DEBUG_DEATH_IF_SUPPORTED(s.front(), "Assertion");
}

TEST(string_pieceDeathTest, back) {
  EXPECT_DEBUG_DEATH_IF_SUPPORTED(string_piece(nullptr).back(), "Assertion");
  EXPECT_DEBUG_DEATH_IF_SUPPORTED(string_piece(nullptr, nullptr).back(),
                                  "Assertion");
  EXPECT_DEBUG_DEATH_IF_SUPPORTED(string_piece("").back(), "Assertion");
  string_piece s("nonempty");
  s.clear();
  EXPECT_DEBUG_DEATH_IF_SUPPORTED(s.back(), "Assertion");
}

TEST(string_piece, substr) {
  string_piece my_string("my really long string");
  EXPECT_EQ("my really long string", my_string.substr(0, string_piece::npos));
  EXPECT_EQ("my really long string", my_string.substr(0));
  EXPECT_EQ("really long string", my_string.substr(3, string_piece::npos));
  EXPECT_EQ("really long string", my_string.substr(3));
  EXPECT_EQ("really", my_string.substr(3, 6));
}

TEST(string_piece, length) {
  EXPECT_EQ(0u, string_piece().size());
  EXPECT_TRUE(string_piece().empty());
  EXPECT_EQ(10u, string_piece("0123456789").size());

  std::string my_string("std::string");
  EXPECT_EQ(my_string.size(), string_piece(my_string).size());
}

TEST(string_piece, clear) {
  string_piece my_string("my really long string");
  EXPECT_EQ("my really long string", my_string);

  string_piece other_string(my_string);
  EXPECT_EQ("my really long string", other_string);

  my_string.clear();
  EXPECT_EQ("", my_string);
  EXPECT_EQ("my really long string", other_string);
}

TEST(string_piece, str) {
  std::string test_string;
  {
    std::string temporary_string("my really long string");
    string_piece my_stringpiece(temporary_string);
    string_piece my_substring = my_stringpiece.substr(3, 6);

    EXPECT_EQ("really", my_substring);
    test_string = my_substring.str();
  }
  EXPECT_EQ("really", test_string);
}

template <char C>
bool find_char(char c) {
  return c == C;
}

TEST(string_piece, find_first_not_matching) {
  string_piece my_string("aaaaaaa b");
  EXPECT_EQ(7u, my_string.find_first_not_matching(find_char<'a'>));
  EXPECT_EQ(0u, my_string.find_first_not_matching(find_char<'b'>));
  EXPECT_EQ(0u, string_piece(" ").find_first_not_matching(::isdigit));
  size_t npos = string_piece::npos;
  EXPECT_EQ(npos, string_piece("").find_first_not_matching(::isdigit));
  EXPECT_EQ(npos, string_piece("123").find_first_not_matching(::isdigit));
  EXPECT_EQ(3u, string_piece("123 ").find_first_not_matching(::isdigit));
}

TEST(string_piece, find_first_not_of) {
  size_t npos = string_piece::npos;
  string_piece my_string("aaaaaaa b");
  EXPECT_EQ(7u, my_string.find_first_not_of("a"));
  EXPECT_EQ(0u, my_string.find_first_not_of("b"));
  EXPECT_EQ(7u, my_string.find_first_not_of('a'));
  EXPECT_EQ(0u, my_string.find_first_not_of('b'));
  EXPECT_EQ(0u, string_piece(" ").find_first_not_of("0123456789"));

  EXPECT_EQ(7u, my_string.find_first_not_of("a", 2));
  EXPECT_EQ(2u, my_string.find_first_not_of("b", 2));
  EXPECT_EQ(7u, my_string.find_first_not_of('a', 2));
  EXPECT_EQ(4u, my_string.find_first_not_of('b', 4));
  EXPECT_EQ(0u, string_piece(" ").find_first_not_of("0123456789"));
  EXPECT_EQ(npos, string_piece(" ").find_first_not_of("0123456789", 5));

  EXPECT_EQ(npos, string_piece("").find_first_not_of("012345689"));
  EXPECT_EQ(npos, string_piece("").find_first_not_of("012345689", 1));
  EXPECT_EQ(npos, string_piece("123").find_first_not_of("0123456789"));
  EXPECT_EQ(npos, string_piece("123").find_first_not_of("0123456789", 1));
  EXPECT_EQ(3u, string_piece("123 ").find_first_not_of("0123456789", 2));
  EXPECT_EQ(npos, string_piece("123 ").find_first_not_of("0123456789", 4));
  EXPECT_EQ(npos, string_piece("").find_first_not_of("1"));
  EXPECT_EQ(npos, string_piece("111").find_first_not_of('1'));
}

TEST(string_piece, find_first_of_char) {
  const size_t npos = string_piece::npos;
  string_piece my_string("my really long string");
  EXPECT_EQ(0u, my_string.find_first_of('m'));
  EXPECT_EQ(3u, my_string.find_first_of('r'));
  EXPECT_EQ(npos, my_string.find_first_of('z'));

  size_t pos = my_string.find_first_of('l');
  EXPECT_EQ(6u, pos);
  // If pos points to a 'l' then we should just find that one
  EXPECT_EQ(6u, my_string.find_first_of('l', pos));
  EXPECT_EQ(7u, my_string.find_first_of('l', pos + 1));
  EXPECT_EQ(10u, my_string.find_first_of('l', pos + 2));
  EXPECT_EQ(npos, my_string.find_first_of('l', pos + 5));
  EXPECT_EQ(npos, my_string.find_first_of('z', 0));
  EXPECT_EQ(npos, my_string.find_first_of('z', npos));

  my_string.clear();
  EXPECT_EQ(npos, my_string.find_first_of('a'));
  EXPECT_EQ(npos, my_string.find_first_of('a', 0));
}

TEST(string_piece, find_first_of) {
  string_piece my_string("aaaaaa b");
  EXPECT_EQ(0u, my_string.find_first_of("a"));
  EXPECT_EQ(7u, my_string.find_first_of("b"));
  EXPECT_EQ(6u, my_string.find_first_of(" "));
  size_t npos = string_piece::npos;
  EXPECT_EQ(npos, my_string.find_first_of("xh"));
  EXPECT_EQ(6u, my_string.find_first_of(" x"));
  EXPECT_EQ(6u, my_string.find_first_of(" b"));
  EXPECT_EQ(0u, my_string.find_first_of("ab"));

  EXPECT_EQ(6u, my_string.find_first_of(" x", 2));
  EXPECT_EQ(6u, my_string.find_first_of(" b", 2));
  EXPECT_EQ(2u, my_string.find_first_of("ab", 2));
  EXPECT_EQ(npos, my_string.find_first_of("ab", 10));

  EXPECT_EQ(npos, my_string.find_first_of("c"));
  EXPECT_EQ(npos, my_string.find_first_of("c", 1));
  EXPECT_EQ(npos, string_piece(" ").find_first_of("a"));
  EXPECT_EQ(npos, string_piece(" ").find_first_of("a", 10));
  EXPECT_EQ(npos, string_piece("aa").find_first_of(""));
  EXPECT_EQ(npos, string_piece("aa").find_first_of("", 1));
  EXPECT_EQ(npos, string_piece("").find_first_of(""));
  EXPECT_EQ(npos, string_piece("").find_first_of("", 1));
  EXPECT_EQ(npos, string_piece("").find_first_of("a"));
  EXPECT_EQ(npos, string_piece("").find_first_of("ae"));
  EXPECT_EQ(npos, string_piece("").find_first_of("ae", 32));
}

TEST(string_piece, find_last_of) {
  string_piece my_string("aaaaaa b");
  EXPECT_EQ(5u, my_string.find_last_of('a'));
  EXPECT_EQ(7u, my_string.find_last_of('b'));
  EXPECT_EQ(6u, my_string.find_last_of(' '));
  EXPECT_EQ(5u, my_string.find_last_of("a"));
  EXPECT_EQ(7u, my_string.find_last_of("b"));
  EXPECT_EQ(6u, my_string.find_last_of(" "));
  size_t npos = string_piece::npos;
  EXPECT_EQ(npos, my_string.find_last_of("xh"));
  EXPECT_EQ(6u, my_string.find_last_of(" x"));
  EXPECT_EQ(7u, my_string.find_last_of(" b"));
  EXPECT_EQ(7u, my_string.find_last_of("ab"));

  EXPECT_EQ(4u, my_string.find_last_of('a', 4));
  EXPECT_EQ(5u, my_string.find_last_of('a', 6));
  EXPECT_EQ(0u, string_piece("abbbaa").find_last_of('a', 3));
  EXPECT_EQ(4u, string_piece("abbbaa").find_last_of('a', 4));
  EXPECT_EQ(5u, string_piece("abbbaa").find_last_of('a', 5));
  EXPECT_EQ(5u, string_piece("abbbaa").find_last_of('a', 6));
  EXPECT_EQ(npos, string_piece("abbbaa").find_last_of('c', 2));

  EXPECT_EQ(npos, my_string.find_last_of("c"));
  EXPECT_EQ(npos, string_piece(" ").find_last_of("a"));
  EXPECT_EQ(npos, string_piece("aa").find_last_of(""));
  EXPECT_EQ(npos, string_piece("").find_last_of(""));
  EXPECT_EQ(npos, string_piece("").find_last_of("a"));
  EXPECT_EQ(npos, my_string.find_last_of('c'));
  EXPECT_EQ(npos, string_piece(" ").find_last_of('a'));
  EXPECT_EQ(npos, string_piece("").find_last_of('a'));
  EXPECT_EQ(npos, string_piece("").find_last_of("ae"));
}

TEST(string_piece, begin_end) {
  const char* my_string = "my really long string";
  string_piece p(my_string);
  size_t pos = 0;
  for (auto it = p.begin(); it != p.end(); ++it) {
    EXPECT_EQ(my_string[pos++], *it);
  }
  pos = 0;
  for (auto c : p) {
    EXPECT_EQ(my_string[pos++], c);
  }
}

TEST(string_piece, front_back) {
  // EXPECT_TRUE() is used here because gtest will think we are comparing
  // between pointer and integer here if EXPECT_EQ() is used.
  const string_piece one_char("a");
  EXPECT_TRUE(one_char.front() == 'a');
  EXPECT_TRUE(one_char.back() == 'a');

  const string_piece two_chars("bc");
  EXPECT_TRUE(two_chars.front() == 'b');
  EXPECT_TRUE(two_chars.back() == 'c');

  const string_piece multi_chars("w   vm g gg t\t");
  EXPECT_TRUE(multi_chars.front() == 'w');
  EXPECT_TRUE(multi_chars.back() == '\t');
}

TEST(string_piece, starts_with) {
  EXPECT_TRUE(string_piece("my string").starts_with("my"));
  EXPECT_TRUE(string_piece("my string").starts_with("my s"));
  EXPECT_TRUE(string_piece("my string").starts_with("m"));
  EXPECT_TRUE(string_piece("my string").starts_with(""));
  EXPECT_TRUE(string_piece("my string").starts_with("my string"));
  EXPECT_TRUE(string_piece("").starts_with(""));

  EXPECT_FALSE(string_piece("").starts_with("a"));
  EXPECT_FALSE(string_piece("my string").starts_with(" "));
  EXPECT_FALSE(string_piece("my string").starts_with("my stq"));
  EXPECT_FALSE(string_piece("my string").starts_with("a"));
  EXPECT_FALSE(string_piece("my string").starts_with("my strings"));
}

TEST(string_piece, find) {
  const size_t npos = string_piece::npos;
  string_piece my_string("gooogle gooogle");

  EXPECT_EQ(0u, my_string.find(""));

  EXPECT_EQ(0u, my_string.find("g"));
  EXPECT_EQ(4u, my_string.find("g", 1));

  EXPECT_EQ(0u, my_string.find("go"));
  EXPECT_EQ(8u, my_string.find("go", 1));

  EXPECT_EQ(1u, my_string.find("oo"));
  EXPECT_EQ(1u, my_string.find("oo", 1));
  EXPECT_EQ(2u, my_string.find("oo", 2));
  EXPECT_EQ(9u, my_string.find("oo", 3));

  EXPECT_EQ(4u, my_string.find("gle"));
  EXPECT_EQ(12u, my_string.find("gle", 5));

  EXPECT_EQ(npos, my_string.find("0"));
  EXPECT_EQ(npos, my_string.find("does-not-exist"));
  EXPECT_EQ(npos, my_string.find("longer than gooogle gooogle"));

  EXPECT_EQ(npos, my_string.find("", npos));
  EXPECT_EQ(npos, my_string.find("gle", npos));
}

TEST(string_piece, get_fields) {
  string_piece input;
  std::vector<string_piece> expected_lines;
  EXPECT_EQ(expected_lines, input.get_fields('\n'));
  EXPECT_EQ(expected_lines, input.get_fields('\n', true));

  input = "first line";
  expected_lines = {"first line"};
  EXPECT_EQ(expected_lines, input.get_fields('\n'));
  EXPECT_EQ(expected_lines, input.get_fields('\n', true));

  input = "first line\n";
  expected_lines = {"first line"};
  EXPECT_EQ(expected_lines, input.get_fields('\n'));
  expected_lines = {"first line\n"};
  EXPECT_EQ(expected_lines, input.get_fields('\n', true));

  input = "\nfirst line";
  expected_lines = {"", "first line"};
  EXPECT_EQ(expected_lines, input.get_fields('\n'));
  expected_lines = {"\n", "first line"};
  EXPECT_EQ(expected_lines, input.get_fields('\n', true));

  input = "first line\nsecond line\nthird line\n";
  expected_lines = {"first line", "second line", "third line"};
  EXPECT_EQ(expected_lines, input.get_fields('\n'));
  expected_lines = {"first line\n", "second line\n", "third line\n"};
  EXPECT_EQ(expected_lines, input.get_fields('\n', true));

  input = "first line\n\nsecond line\n\nthird line\n\n";
  expected_lines = {"first line", "", "second line", "", "third line", ""};
  EXPECT_EQ(expected_lines, input.get_fields('\n'));
  expected_lines = {"first line\n", "\n",           "second line\n",
                    "\n",           "third line\n", "\n"};
  EXPECT_EQ(expected_lines, input.get_fields('\n', true));
}

TEST(string_piece, operator_stream_out) {
  std::stringstream stream;
  string_piece my_string("my really long string");
  stream << my_string;
  EXPECT_EQ("my really long string", stream.str());
  stream.str("");
  stream << my_string.substr(3, 6);
  EXPECT_EQ("really", stream.str());
  stream.str("");
  stream << string_piece();
  EXPECT_EQ("", stream.str());
}

TEST(string_piece, lrstrip) {
  string_piece nothing_to_remove("abcdefg");
  EXPECT_EQ("abcdefg", nothing_to_remove.lstrip("hijklmn"));
  EXPECT_EQ("abcdefg", nothing_to_remove.rstrip("hijklmn"));
  EXPECT_EQ("abcdefg", nothing_to_remove.strip("hijklmn"));

  string_piece empty_string("");
  EXPECT_EQ(0u, empty_string.lstrip("google").size());
  EXPECT_EQ(0u, empty_string.rstrip("google").size());
  EXPECT_EQ(0u, empty_string.strip("google").size());

  string_piece remove_nothing("asdfghjkl");
  EXPECT_EQ("asdfghjkl", remove_nothing.lstrip(""));
  EXPECT_EQ("asdfghjkl", remove_nothing.rstrip(""));
  EXPECT_EQ("asdfghjkl", remove_nothing.strip(""));

  string_piece strip_numbers("0123g4o5o6g7l8e9");
  EXPECT_EQ("g4o5o6g7l8e9", strip_numbers.lstrip("0123456789"));
  EXPECT_EQ("0123g4o5o6g7l8e", strip_numbers.rstrip("0123456789"));
  EXPECT_EQ("g4o5o6g7l8e", strip_numbers.strip("0123456789"));
}

TEST(string_piece, strip_whitespace) {
  string_piece lots_of_space("  b i n g o      ");
  EXPECT_EQ("b i n g o", lots_of_space.strip_whitespace());

  string_piece whitespaces("\v\t\f\n\rleft\r\t\f\n\vright\f\n\t\v\r");
  EXPECT_EQ("left\r\t\f\n\vright", whitespaces.strip_whitespace());

  string_piece remove_all("  \t  ");
  EXPECT_EQ(0u, remove_all.strip_whitespace().size());
}

TEST(string_piece, not_equal) {
  EXPECT_FALSE(string_piece() != string_piece());
  EXPECT_FALSE(string_piece("") != string_piece());
  EXPECT_TRUE(string_piece() != string_piece(" "));
  EXPECT_FALSE(string_piece("abc") != string_piece("abc"));
  EXPECT_TRUE(string_piece("abc") != string_piece("abc "));
  EXPECT_TRUE(string_piece("abc") != string_piece("abd"));

  EXPECT_FALSE("" != string_piece());
  EXPECT_FALSE("" != string_piece(""));
  EXPECT_TRUE(" " != string_piece(""));
  EXPECT_FALSE("abc" != string_piece("abc"));
  EXPECT_TRUE(" abc" != string_piece("abc"));
  EXPECT_TRUE("abd" != string_piece("abc"));
}

TEST(string_piece, data) {
  EXPECT_EQ(nullptr, string_piece().data());
  const char* empty = "";
  EXPECT_EQ(empty, string_piece(empty).data());
  const char* space = " ";
  EXPECT_EQ(space, string_piece(space).data());
  const char* a = "a";
  EXPECT_EQ(a, string_piece(a).data());
  const char* abc = "abc";
  EXPECT_EQ(abc, string_piece(abc).data());
  EXPECT_EQ(abc + 1, string_piece(abc).substr(1).data());
  EXPECT_EQ(abc + 3, string_piece(abc).substr(3).data());
}

TEST(string_piece, unordered_map) {
  std::unordered_map<string_piece, int> dict;
  dict["abc"] = 123;
  EXPECT_EQ(123, dict["abc"]);
}

}  // anonymous namespace
