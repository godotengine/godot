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

#include "libshaderc_util/version_profile.h"

#include <cctype>
#include <sstream>

namespace {

const int kVersionNumberLength = 3;
const int kMaxProfileLength = 13;  // strlen(compatibility)
const int kMaxVersionProfileLength = kVersionNumberLength + kMaxProfileLength;
const int kMinVersionProfileLength = kVersionNumberLength;

}  // anonymous namespace

namespace shaderc_util {

bool ParseVersionProfile(const std::string& version_profile, int* version,
                         EProfile* profile) {
  if (version_profile.size() < kMinVersionProfileLength ||
      version_profile.size() > kMaxVersionProfileLength ||
      !::isdigit(version_profile.front()))
    return false;

  std::string profile_string;
  std::istringstream(version_profile) >> *version >> profile_string;

  if (!IsKnownVersion(*version)) {
    return false;
  }
  if (profile_string.empty()) {
    *profile = ENoProfile;
  } else if (profile_string == "core") {
    *profile = ECoreProfile;
  } else if (profile_string == "es") {
    *profile = EEsProfile;
  } else if (profile_string == "compatibility") {
    *profile = ECompatibilityProfile;
  } else {
    return false;
  }

  return true;
}

}  // namespace shaderc_util
