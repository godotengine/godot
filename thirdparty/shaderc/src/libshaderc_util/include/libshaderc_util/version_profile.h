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

#ifndef LIBSHADERC_UTIL_INC_VERSION_PROFILE_H_
#define LIBSHADERC_UTIL_INC_VERSION_PROFILE_H_

#include <string>

#include "glslang/MachineIndependent/Versions.h"

namespace shaderc_util {

// Returns true if the given version is an accepted GLSL (ES) version.
inline bool IsKnownVersion(int version) {
  switch (version) {
    case 100:
    case 110:
    case 120:
    case 130:
    case 140:
    case 150:
    case 300:
    case 310:
    case 320:
    case 330:
    case 400:
    case 410:
    case 420:
    case 430:
    case 440:
    case 450:
    case 460:
      return true;
    default:
      break;
  }
  return false;
}

// Given a string version_profile containing both version and profile, decodes
// it and puts the decoded version in version, decoded profile in profile.
// Returns true if decoding is successful and version and profile are accepted.
// This does not validate the version number against the profile.  For example,
// "460es" doesn't make sense (yet), but is still accepted.
bool ParseVersionProfile(const std::string& version_profile, int* version,
                         EProfile* profile);

}  // namespace shaderc_util

#endif  // LIBSHADERC_UTIL_INC_VERSION_PROFILE_H_
