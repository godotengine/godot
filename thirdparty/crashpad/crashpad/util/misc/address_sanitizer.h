// Copyright 2016 The Crashpad Authors. All rights reserved.
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

#ifndef CRASHPAD_UTIL_MISC_ADDRESS_SANITIZER_H_
#define CRASHPAD_UTIL_MISC_ADDRESS_SANITIZER_H_

#include "base/compiler_specific.h"
#include "build/build_config.h"

#if !defined(ADDRESS_SANITIZER)
#if HAS_FEATURE(address_sanitizer) || \
    (defined(COMPILER_GCC) && defined(__SANITIZE_ADDRESS__))
#define ADDRESS_SANITIZER 1
#endif
#endif  // !defined(ADDRESS_SANITIZER)

#endif  // CRASHPAD_UTIL_MISC_ADDRESS_SANITIZER_H_
