// Copyright 2018 The Effcee Authors.
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

#ifndef EFFCEE_TO_STRING_H
#define EFFCEE_TO_STRING_H

#include <string>
#include "effcee.h"

namespace effcee {

// Returns a copy of a StringPiece, as a std::string.
inline std::string ToString(effcee::StringPiece s) {
  return std::string(s.data(), s.size());
}
}  // namespace effcee

#endif
