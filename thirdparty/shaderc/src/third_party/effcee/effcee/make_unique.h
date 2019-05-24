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

#ifndef EFFCEE_MAKE_UNIQUE_H
#define EFFCEE_MAKE_UNIQUE_H

#include <memory>

namespace effcee {

// Constructs an object of type T and wraps it in a std::unique_ptr.
// This functionality comes with C++14.  The following is the standard
// recipe for use with C++11.
template <typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

}  // namespace effcee

#endif
