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

#include "effcee.h"

namespace {

using effcee::Result;
using ::testing::Combine;
using ::testing::Eq;
using ::testing::Not;
using ::testing::ValuesIn;

using Status = effcee::Result::Status;

// Result class

// Returns a vector of all failure status values.
std::vector<Status> AllFailStatusValues() {
  return {Status::NoRules, Status::BadRule};
}

// Returns a vector of all status values.
std::vector<Status> AllStatusValues() {
  auto result = AllFailStatusValues();
  result.push_back(Status::Ok);
  return result;
}

// Test one-argument constructor.

using ResultStatusTest = ::testing::TestWithParam<Status>;

TEST_P(ResultStatusTest, ConstructWithAnyStatus) {
  Result result(GetParam());
  EXPECT_THAT(result.status(), Eq(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(AllStatus, ResultStatusTest,
                         ValuesIn(AllStatusValues()));

// Test two-argument constructor.

using ResultStatusMessageCase = std::tuple<Status, std::string>;

using ResultStatusMessageTest =
    ::testing::TestWithParam<ResultStatusMessageCase>;

TEST_P(ResultStatusMessageTest, ConstructWithStatusAndMessage) {
  Result result(std::get<0>(GetParam()), std::get<1>(GetParam()));
  EXPECT_THAT(result.status(), Eq(std::get<0>(GetParam())));
  EXPECT_THAT(result.message(), Eq(std::get<1>(GetParam())));
}

INSTANTIATE_TEST_SUITE_P(SampleStatusAndMessage, ResultStatusMessageTest,
                         Combine(ValuesIn(AllStatusValues()),
                                 ValuesIn(std::vector<std::string>{
                                     "", "foo bar", "and, how!\n"})));

TEST(ResultConversionTest, OkStatusConvertsToTrue) {
  Result result(Status::Ok);
  bool as_bool = result;
  EXPECT_THAT(as_bool, Eq(true));
}

// Test conversion to bool.

using ResultFailConversionTest = ::testing::TestWithParam<Status>;

TEST_P(ResultFailConversionTest, AnyFailStatusConvertsToFalse) {
  Result result(GetParam());
  bool as_bool = result;
  EXPECT_THAT(as_bool, Eq(false));
}

INSTANTIATE_TEST_SUITE_P(FailStatus, ResultFailConversionTest,
                         ValuesIn(AllFailStatusValues()));

TEST(ResultMessage, SetMessageReturnsSelf) {
  Result result(Status::Ok);
  Result& other = result.SetMessage("");
  EXPECT_THAT(&other, Eq(&result));
}

TEST(ResultMessage, MessageDefaultsToEmpty) {
  Result result(Status::Ok);
  EXPECT_THAT(result.message(), Eq(""));
}

TEST(ResultMessage, SetMessageOnceSetsMessage) {
  Result result(Status::Ok);
  result.SetMessage("foo");
  EXPECT_THAT(result.message(), Eq("foo"));
}

TEST(ResultMessage, SetMessageCopiesString) {
  Result result(Status::Ok);
  std::string original("foo");
  result.SetMessage(original);
  EXPECT_THAT(result.message().data(), Not(Eq(original.data())));
}

TEST(ResultMessage, SetMessageEmtpyStringPossible) {
  Result result(Status::Ok);
  result.SetMessage("");
  EXPECT_THAT(result.message(), Eq(""));
}

TEST(ResultMessage, SetMessageTwiceRetainsLastMessage) {
  Result result(Status::Ok);
  result.SetMessage("foo");
  result.SetMessage("bar baz");
  EXPECT_THAT(result.message(), Eq("bar baz"));
}

}  // namespace
