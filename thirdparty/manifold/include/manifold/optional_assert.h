// Copyright 2022 The Manifold Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#ifdef MANIFOLD_EXCEPTIONS
#include <stdexcept>
#endif

#ifdef MANIFOLD_DEBUG
#include <iostream>
#include <sstream>
#include <string>

template <typename Ex>
void Assert(bool condition, const char* file, int line, const std::string& cond,
            const std::string& msg) {
  if (!condition) {
    std::ostringstream output;
    output << "Error in file: " << file << " (" << line << "): \'" << cond
           << "\' is false: " << msg;
    throw Ex(output.str());
  }
}
#define DEBUG_ASSERT(condition, EX, msg) \
  Assert<EX>(condition, __FILE__, __LINE__, #condition, msg);
#else
#define DEBUG_ASSERT(condition, EX, msg)
#endif

#ifdef MANIFOLD_EXCEPTIONS
#define ASSERT(condition, EX) \
  if (!(condition)) throw(EX);
#else
#define ASSERT(condition, EX)
#endif
