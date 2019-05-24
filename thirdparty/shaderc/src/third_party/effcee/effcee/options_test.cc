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

#include "effcee.h"

namespace {

using effcee::Options;
using ::testing::Eq;
using ::testing::Not;

// Options class

// Prefix property

TEST(Options, DefaultPrefixIsCHECK) {
  EXPECT_THAT(Options().prefix(), "CHECK");
}

TEST(Options, SetPrefixReturnsSelf) {
  Options options;
  const Options& other = options.SetPrefix("");
  EXPECT_THAT(&other, &options);
}

TEST(Options, SetPrefixOnceSetsPrefix) {
  Options options;
  options.SetPrefix("foo");
  EXPECT_THAT(options.prefix(), Eq("foo"));
}

TEST(Options, SetPrefixCopiesString) {
  Options options;
  std::string original("foo");
  options.SetPrefix(original);
  EXPECT_THAT(options.prefix().data(), Not(Eq(original.data())));
}

TEST(Options, SetPrefixEmptyStringPossible) {
  Options options;
  // This is not useful.
  options.SetPrefix("");
  EXPECT_THAT(options.prefix(), Eq(""));
}

TEST(Options, SetPrefixTwiceRetainsLastPrefix) {
  Options options;
  options.SetPrefix("foo");
  options.SetPrefix("bar baz");
  EXPECT_THAT(options.prefix(), Eq("bar baz"));
}


// Input name property

TEST(Options, DefaultInputNameIsStdin) {
  EXPECT_THAT(Options().input_name(), "<stdin>");
}

TEST(Options, SetInputNameReturnsSelf) {
  Options options;
  const Options& other = options.SetInputName("");
  EXPECT_THAT(&other, &options);
}

TEST(Options, SetInputNameOnceSetsInputName) {
  Options options;
  options.SetInputName("foo");
  EXPECT_THAT(options.input_name(), Eq("foo"));
}

TEST(Options, SetInputNameCopiesString) {
  Options options;
  std::string original("foo");
  options.SetInputName(original);
  EXPECT_THAT(options.input_name().data(), Not(Eq(original.data())));
}

TEST(Options, SetInputNameEmptyStringPossible) {
  Options options;
  options.SetInputName("");
  EXPECT_THAT(options.input_name(), Eq(""));
}

TEST(Options, SetInputNameTwiceRetainsLastInputName) {
  Options options;
  options.SetInputName("foo");
  options.SetInputName("bar baz");
  EXPECT_THAT(options.input_name(), Eq("bar baz"));
}

// Checks name property

TEST(Options, DefaultChecksNameIsStdin) {
  EXPECT_THAT(Options().checks_name(), "<stdin>");
}

TEST(Options, SetChecksNameReturnsSelf) {
  Options options;
  const Options& other = options.SetChecksName("");
  EXPECT_THAT(&other, &options);
}

TEST(Options, SetChecksNameOnceSetsChecksName) {
  Options options;
  options.SetChecksName("foo");
  EXPECT_THAT(options.checks_name(), Eq("foo"));
}

TEST(Options, SetChecksNameCopiesString) {
  Options options;
  std::string original("foo");
  options.SetChecksName(original);
  EXPECT_THAT(options.checks_name().data(), Not(Eq(original.data())));
}

TEST(Options, SetChecksNameEmptyStringPossible) {
  Options options;
  options.SetChecksName("");
  EXPECT_THAT(options.checks_name(), Eq(""));
}

TEST(Options, SetChecksNameTwiceRetainsLastChecksName) {
  Options options;
  options.SetChecksName("foo");
  options.SetChecksName("bar baz");
  EXPECT_THAT(options.checks_name(), Eq("bar baz"));
}

}  // namespace
