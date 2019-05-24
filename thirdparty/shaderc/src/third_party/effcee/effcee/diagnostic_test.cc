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

#include "diagnostic.h"

namespace {

using effcee::Diagnostic;
using effcee::Result;
using testing::Eq;

using Status = effcee::Result::Status;

// Check conversion preserves status.
TEST(Diagnostic, ConvertsToResultWithSameOkStatus) {
  const Diagnostic d(Status::Ok);
  const Result r(d);
  EXPECT_THAT(r.status(), Eq(Status::Ok));
}

TEST(Diagnostic, ConvertsToResultWithSameFailStatus) {
  const Diagnostic d(Status::Fail);
  const Result r(d);
  EXPECT_THAT(r.status(), Eq(Status::Fail));
}

// Check conversion, with messages.

TEST(Diagnostic, MessageDefaultsToEmpty) {
  const Diagnostic d(Status::Ok);
  const Result r(d);
  EXPECT_THAT(r.message(), Eq(""));
}

TEST(Diagnostic, MessageAccumulatesValuesOfDifferentTypes) {
  Diagnostic d(Status::Ok);
  d << "hello" << ' ' << 42 << " and " << 32u << " and " << 1.25;
  const Result r(d);
  EXPECT_THAT(r.message(), Eq("hello 42 and 32 and 1.25"));
}

// Check copying

TEST(Diagnostic, CopyRetainsOriginalMessage) {
  Diagnostic d(Status::Ok);
  d << "hello";
  Diagnostic d2 = d;
  const Result r(d2);
  EXPECT_THAT(r.message(), Eq("hello"));
}

TEST(Diagnostic, ShiftOnCopyAppendsToOriginalMessage) {
  Diagnostic d(Status::Ok);
  d << "hello";
  Diagnostic d2 = d;
  d2 << " world";
  const Result r(d2);
  EXPECT_THAT(r.message(), Eq("hello world"));
}


}  // namespace
