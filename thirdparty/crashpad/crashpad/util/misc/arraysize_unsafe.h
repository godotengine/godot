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

#ifndef CRASHPAD_UTIL_MISC_ARRAYSIZE_UNSAFE_H_
#define CRASHPAD_UTIL_MISC_ARRAYSIZE_UNSAFE_H_

//! \file

//! \brief Not the safest way of computing an array’s size…
//!
//! `#%include "base/macros.h"` and use its `arraysize()` instead. This macro
//! should only be used in rare situations where `arraysize()` does not
//! function.
#define ARRAYSIZE_UNSAFE(array) (sizeof(array) / sizeof(array[0]))

#endif  // CRASHPAD_UTIL_MISC_ARRAYSIZE_UNSAFE_H_
