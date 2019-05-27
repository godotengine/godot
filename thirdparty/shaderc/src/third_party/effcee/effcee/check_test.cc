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

#include <vector>
#include "gmock/gmock.h"

#include "check.h"

namespace {

using effcee::Check;
using effcee::Options;
using effcee::CheckList;
using effcee::ParseChecks;
using effcee::Result;
using effcee::StringPiece;
using ::testing::Combine;
using ::testing::Eq;
using ::testing::HasSubstr;
using ::testing::ValuesIn;

using Part = effcee::Check::Part;
using Status = effcee::Result::Status;
using Type = Check::Type;
using VarMapping = effcee::VarMapping;

// Check class

// Returns a vector of all Check types.
std::vector<Type> AllTypes() {
  return {Type::Simple, Type::Next,  Type::Same,
          Type::DAG,    Type::Label, Type::Not};
}

using CheckTypeTest = ::testing::TestWithParam<Type>;

TEST_P(CheckTypeTest, ConstructWithAnyType) {
  Check check(GetParam(), "");
  EXPECT_THAT(check.type(), Eq(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(AllTypes, CheckTypeTest, ValuesIn(AllTypes()));

using CheckParamTest = ::testing::TestWithParam<StringPiece>;

TEST_P(CheckParamTest, ConstructWithSampleParamValue) {
  Check check(Type::Simple, GetParam());
  // The contents are the same.
  EXPECT_THAT(check.param(), Eq(GetParam()));
  // The referenced storage is the same.
  EXPECT_THAT(check.param().data(), Eq(GetParam().data()));
}

INSTANTIATE_TEST_SUITE_P(SampleParams, CheckParamTest,
                         ValuesIn(std::vector<StringPiece>{
                             "", "a b c", "The wind {{in}} the willows\n",
                             "Bring me back to the mountains of yore."}));

// Equality operator
TEST(CheckEqualityTest, TrueWhenAllComponentsSame) {
  EXPECT_TRUE(Check(Type::Simple, "abc") == Check(Type::Simple, "abc"));
}

TEST(CheckEqualityTest, FalseWhenTypeDifferent) {
  EXPECT_FALSE(Check(Type::Simple, "abc") == Check(Type::Next, "abc"));
}

TEST(CheckEqualityTest, FalseWhenParamDifferent) {
  EXPECT_FALSE(Check(Type::Simple, "abc") == Check(Type::Simple, "def"));
}

// Inequality operator
TEST(CheckInequalityTest, FalseWhenAllComponentsSame) {
  EXPECT_FALSE(Check(Type::Simple, "abc") != Check(Type::Simple, "abc"));
}

TEST(CheckInequalityTest, TrueWhenTypeDifferent) {
  EXPECT_TRUE(Check(Type::Simple, "abc") != Check(Type::Next, "abc"));
}

TEST(CheckInequalityTest, TrueWhenParamDifferent) {
  EXPECT_TRUE(Check(Type::Simple, "abc") != Check(Type::Simple, "def"));
}

// ParseChecks free function

TEST(ParseChecks, FreeFunctionLinks) {
  std::pair<Result, CheckList> parsed(ParseChecks("", Options()));
}

TEST(ParseChecks, FailWhenRulePrefixIsEmpty) {
  const auto parsed(ParseChecks("CHECK: now", Options().SetPrefix("")));
  const Result& result = parsed.first;
  const CheckList& pattern = parsed.second;
  EXPECT_THAT(result.status(), Eq(Status::BadOption));
  EXPECT_THAT(result.message(), Eq("Rule prefix is empty"));
  EXPECT_THAT(pattern.size(), Eq(0));
}

TEST(ParseChecks, FailWhenRulePrefixIsWhitespace) {
  const auto parsed(ParseChecks("CHECK: now", Options().SetPrefix("\t\n ")));
  const Result& result = parsed.first;
  const CheckList& pattern = parsed.second;
  EXPECT_THAT(result.status(), Eq(Status::BadOption));
  EXPECT_THAT(result.message(),
              Eq("Rule prefix is whitespace.  That's silly."));
  EXPECT_THAT(pattern.size(), Eq(0));
}

TEST(ParseChecks, FailWhenChecksAbsent) {
  const auto parsed(ParseChecks("no checks", Options()));
  const Result& result = parsed.first;
  const CheckList& pattern = parsed.second;
  EXPECT_THAT(result.status(), Eq(Status::NoRules));
  EXPECT_THAT(result.message(),
              Eq("No check rules specified. Looking for prefix CHECK"));
  EXPECT_THAT(pattern.size(), Eq(0));
}

TEST(ParseChecks, FailWhenChecksAbsentWithCustomPrefix) {
  const auto parsed(ParseChecks("CHECK: now", Options().SetPrefix("FOO")));
  const Result& result = parsed.first;
  const CheckList& pattern = parsed.second;
  EXPECT_THAT(result.status(), Eq(Status::NoRules));
  EXPECT_THAT(result.message(),
              Eq("No check rules specified. Looking for prefix FOO"));
  EXPECT_THAT(pattern.size(), Eq(0));
}

TEST(ParseChecks, FindSimpleCheck) {
  const auto parsed = ParseChecks("CHECK: now", Options());
  EXPECT_THAT(parsed.first.status(), Eq(Status::Ok));
  EXPECT_THAT(parsed.second, Eq(CheckList({Check(Type::Simple, "now")})));
}

TEST(ParseChecks, FindSimpleCheckWithCustomPrefix) {
  const auto parsed = ParseChecks("FOO: how", Options().SetPrefix("FOO"));
  EXPECT_THAT(parsed.first.status(), Eq(Status::Ok));
  EXPECT_THAT(parsed.second, Eq(CheckList({Check(Type::Simple, "how")})));
}

TEST(ParseChecks, FindSimpleCheckWithCustomPrefixHavingRegexpMetachars) {
  const auto parsed = ParseChecks("[::alpha::]^\\d: how",
                                  Options().SetPrefix("[::alpha::]^\\d"));
  EXPECT_THAT(parsed.first.status(), Eq(Status::Ok));
  EXPECT_THAT(parsed.second, Eq(CheckList({Check(Type::Simple, "how")})));
}

TEST(ParseChecks, FindSimpleCheckPartwayThroughLine) {
  const auto parsed = ParseChecks("some other garbageCHECK: now", Options());
  EXPECT_THAT(parsed.first.status(), Eq(Status::Ok));
  EXPECT_THAT(parsed.second, Eq(CheckList({Check(Type::Simple, "now")})));
}

TEST(ParseChecks, FindSimpleCheckCheckListWithoutSurroundingWhitespace) {
  const auto parsed = ParseChecks("CHECK:now", Options());
  EXPECT_THAT(parsed.first.status(), Eq(Status::Ok));
  EXPECT_THAT(parsed.second, Eq(CheckList({Check(Type::Simple, "now")})));
}

TEST(ParseChecks, FindSimpleCheckCheckListWhileStrippingSurroundingWhitespace) {
  const auto parsed = ParseChecks("CHECK: \t   now\t\t  ", Options());
  EXPECT_THAT(parsed.first.status(), Eq(Status::Ok));
  EXPECT_THAT(parsed.second, Eq(CheckList({Check(Type::Simple, "now")})));
}

TEST(ParseChecks, FindSimpleCheckCountsLinesCorrectly) {
  const auto parsed = ParseChecks("\n\nCHECK: now", Options());
  EXPECT_THAT(parsed.first.status(), Eq(Status::Ok));
  EXPECT_THAT(parsed.second, Eq(CheckList({Check(Type::Simple, "now")})));
}

TEST(ParseChecks, FindSimpleChecksOnSeparateLines) {
  const auto parsed =
      ParseChecks("CHECK: now\n\n\nCHECK: and \n CHECK: then", Options());
  EXPECT_THAT(parsed.first.status(), Eq(Status::Ok));
  EXPECT_THAT(parsed.second, Eq(CheckList({Check(Type::Simple, "now"),
                                           Check(Type::Simple, "and"),
                                           Check(Type::Simple, "then")})));
}

TEST(ParseChecks, FindSimpleChecksOnlyOncePerLine) {
  const auto parsed = ParseChecks("CHECK: now CHECK: then", Options());
  EXPECT_THAT(parsed.first.status(), Eq(Status::Ok));
  EXPECT_THAT(parsed.second,
              Eq(CheckList({Check(Type::Simple, "now CHECK: then")})));
}

// Test parsing of the different check rule types.

using ParseChecksTypeTest = ::testing::TestWithParam<
    std::tuple<std::string, std::pair<std::string, Type>>>;

TEST_P(ParseChecksTypeTest, Successful) {
  const auto& prefix = std::get<0>(GetParam());
  const auto& type_str = std::get<0>(std::get<1>(GetParam()));
  const Type& type = std::get<1>(std::get<1>(GetParam()));
  // A CHECK-SAME rule can't appear first, so insert a CHECK: rule first.
  const std::string input = prefix + ": here\n" + prefix + type_str + ": now";
  const auto parsed = ParseChecks(input, Options().SetPrefix(prefix));
  EXPECT_THAT(parsed.first.status(), Eq(Status::Ok));
  EXPECT_THAT(parsed.second,
              Eq(CheckList({Check(Type::Simple, "here"), Check(type, "now")})));
}

// Returns a vector of pairs. Each pair has first member being a check type
// suffix, and the second member is the corresponding check type.
std::vector<std::pair<std::string, Type>> AllCheckTypesAsPairs() {
  return {
      {"", Type::Simple},  {"-NEXT", Type::Next},   {"-SAME", Type::Same},
      {"-DAG", Type::DAG}, {"-LABEL", Type::Label}, {"-NOT", Type::Not},
  };
}

INSTANTIATE_TEST_SUITE_P(AllCheckTypes, ParseChecksTypeTest,
                         Combine(ValuesIn(std::vector<std::string>{"CHECK",
                                                                   "FOO"}),
                                 ValuesIn(AllCheckTypesAsPairs())));

using ParseChecksTypeFailTest = ::testing::TestWithParam<
    std::tuple<std::string, std::pair<std::string, Type>>>;

// This is just one way to fail.
TEST_P(ParseChecksTypeFailTest, FailureWhenNoColon) {
  const auto& prefix = std::get<0>(GetParam());
  const auto& type_str = std::get<0>(std::get<1>(GetParam()));
  const std::string input = prefix + type_str + "BAD now";
  const auto parsed = ParseChecks(input, Options().SetPrefix(prefix));
  EXPECT_THAT(parsed.first.status(), Eq(Status::NoRules));
  EXPECT_THAT(parsed.second, Eq(CheckList{}));
}

INSTANTIATE_TEST_SUITE_P(AllCheckTypes, ParseChecksTypeFailTest,
                         Combine(ValuesIn(std::vector<std::string>{"CHECK",
                                                                   "FOO"}),
                                 ValuesIn(AllCheckTypesAsPairs())));

TEST(ParseChecks, CheckSameCantBeFirst) {
  const auto parsed = ParseChecks("CHECK-SAME: now", Options());
  EXPECT_THAT(parsed.first.status(), Eq(Status::BadRule));
  EXPECT_THAT(parsed.first.message(),
              HasSubstr("CHECK-SAME can't be the first check rule"));
  EXPECT_THAT(parsed.second, Eq(CheckList({})));
}

TEST(ParseChecks, CheckSameCantBeFirstDifferentPrefix) {
  const auto parsed = ParseChecks("BOO-SAME: now", Options().SetPrefix("BOO"));
  EXPECT_THAT(parsed.first.status(), Eq(Status::BadRule));
  EXPECT_THAT(parsed.first.message(),
              HasSubstr("BOO-SAME can't be the first check rule"));
  EXPECT_THAT(parsed.second, Eq(CheckList({})));
}

// Check::Matches
struct CheckMatchCase {
  std::string input;
  Check check;
  bool expected;
  std::string remaining;
  std::string captured;
};

using CheckMatchTest = ::testing::TestWithParam<CheckMatchCase>;

TEST_P(CheckMatchTest, Samples) {
  StringPiece str = GetParam().input;
  StringPiece captured;
  VarMapping vars;
  const bool matched = GetParam().check.Matches(&str, &captured, &vars);
  EXPECT_THAT(matched, Eq(GetParam().expected))
      << "Failed on input " << GetParam().input;
  EXPECT_THAT(std::string(str.data(), str.size()), Eq(GetParam().remaining));
  EXPECT_THAT(std::string(captured.data(), captured.size()),
              Eq(GetParam().captured));
  EXPECT_TRUE(vars.empty());
}

INSTANTIATE_TEST_SUITE_P(
    Simple, CheckMatchTest,
    ValuesIn(std::vector<CheckMatchCase>{
        {"hello", Check(Type::Simple, "hello"), true, "", "hello"},
        {"world", Check(Type::Simple, "hello"), false, "world", ""},
        {"in hello now", Check(Type::Simple, "hello"), true, " now", "hello"},
        {"hello", Check(Type::Same, "hello"), true, "", "hello"},
        {"world", Check(Type::Same, "hello"), false, "world", ""},
        {"in hello now", Check(Type::Same, "hello"), true, " now", "hello"},
        {"hello", Check(Type::Next, "hello"), true, "", "hello"},
        {"world", Check(Type::Next, "hello"), false, "world", ""},
        {"in hello now", Check(Type::Next, "hello"), true, " now", "hello"},
        {"hello", Check(Type::DAG, "hello"), true, "", "hello"},
        {"world", Check(Type::DAG, "hello"), false, "world", ""},
        {"in hello now", Check(Type::DAG, "hello"), true, " now", "hello"},
        {"hello", Check(Type::Label, "hello"), true, "", "hello"},
        {"world", Check(Type::Label, "hello"), false, "world", ""},
        {"in hello now", Check(Type::Label, "hello"), true, " now", "hello"},
        {"hello", Check(Type::Label, "hello"), true, "", "hello"},
        {"world", Check(Type::Label, "hello"), false, "world", ""},
        {"in hello now", Check(Type::Label, "hello"), true, " now", "hello"},
        {"hello", Check(Type::Not, "hello"), true, "", "hello"},
        {"world", Check(Type::Not, "hello"), false, "world", ""},
        {"in hello now", Check(Type::Not, "hello"), true, " now", "hello"},
    }));

// Check::Part::Regex

TEST(CheckPart, FixedPartRegex) {
  VarMapping vm;
  EXPECT_THAT(Part(Part::Type::Fixed, "abc").Regex(vm), Eq("abc"));
  EXPECT_THAT(Part(Part::Type::Fixed, "a.bc").Regex(vm), Eq("a\\.bc"));
  EXPECT_THAT(Part(Part::Type::Fixed, "a?bc").Regex(vm), Eq("a\\?bc"));
  EXPECT_THAT(Part(Part::Type::Fixed, "a+bc").Regex(vm), Eq("a\\+bc"));
  EXPECT_THAT(Part(Part::Type::Fixed, "a*bc").Regex(vm), Eq("a\\*bc"));
  EXPECT_THAT(Part(Part::Type::Fixed, "a[b]").Regex(vm), Eq("a\\[b\\]"));
  EXPECT_THAT(Part(Part::Type::Fixed, "a[-]").Regex(vm), Eq("a\\[\\-\\]"));
  EXPECT_THAT(Part(Part::Type::Fixed, "a(-)b").Regex(vm), Eq("a\\(\\-\\)b"));
}

TEST(CheckPart, RegexPartRegex) {
  VarMapping vm;
  EXPECT_THAT(Part(Part::Type::Regex, "abc").Regex(vm), Eq("abc"));
  EXPECT_THAT(Part(Part::Type::Regex, "a.bc").Regex(vm), Eq("a.bc"));
  EXPECT_THAT(Part(Part::Type::Regex, "a?bc").Regex(vm), Eq("a?bc"));
  EXPECT_THAT(Part(Part::Type::Regex, "a+bc").Regex(vm), Eq("a+bc"));
  EXPECT_THAT(Part(Part::Type::Regex, "a*bc").Regex(vm), Eq("a*bc"));
  EXPECT_THAT(Part(Part::Type::Regex, "a[b]").Regex(vm), Eq("a[b]"));
  EXPECT_THAT(Part(Part::Type::Regex, "a[-]").Regex(vm), Eq("a[-]"));
  EXPECT_THAT(Part(Part::Type::Regex, "a(-)b").Regex(vm), Eq("a(-)b"));
}

}  // namespace

