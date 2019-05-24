// Copyright (c) 2015-2016 Google Inc.
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

#include <sstream>
#include <string>

#include "gmock/gmock.h"
#include "test/unit_spirv.h"

namespace spvtools {
namespace {

using ::testing::AnyOf;
using ::testing::Eq;
using ::testing::Ge;
using ::testing::StartsWith;

void CheckFormOfHighLevelVersion(const std::string& version) {
  std::istringstream s(version);
  char v = 'x';
  int year = -1;
  char period = 'x';
  int index = -1;
  s >> v >> year >> period >> index;
  EXPECT_THAT(v, Eq('v'));
  EXPECT_THAT(year, Ge(2016));
  EXPECT_THAT(period, Eq('.'));
  EXPECT_THAT(index, Ge(0));
  EXPECT_TRUE(s.good() || s.eof());

  std::string rest;
  s >> rest;
  EXPECT_THAT(rest, AnyOf("", "-dev"));
}

TEST(SoftwareVersion, ShortIsCorrectForm) {
  SCOPED_TRACE("short form");
  CheckFormOfHighLevelVersion(spvSoftwareVersionString());
}

TEST(SoftwareVersion, DetailedIsCorrectForm) {
  const std::string detailed_version(spvSoftwareVersionDetailsString());
  EXPECT_THAT(detailed_version, StartsWith("SPIRV-Tools v"));

  // Parse the high level version.
  const std::string from_v =
      detailed_version.substr(detailed_version.find_first_of('v'));
  const size_t first_space_after_v_or_npos = from_v.find_first_of(' ');
  SCOPED_TRACE(detailed_version);
  CheckFormOfHighLevelVersion(from_v.substr(0, first_space_after_v_or_npos));

  // We don't actually care about what comes after the version number.
}

}  // namespace
}  // namespace spvtools
