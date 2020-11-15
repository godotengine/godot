// Copyright 2017 The Crashpad Authors. All rights reserved.
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

#ifndef CRASHPAD_THIRD_PARTY_ZLIB_ZLIB_CRASHPAD_H_
#define CRASHPAD_THIRD_PARTY_ZLIB_ZLIB_CRASHPAD_H_

// #include this file instead of the system version of <zlib.h> or equivalent
// available at any other location in the source tree. It will #include the
// proper <zlib.h> depending on how the build has been configured.

#if defined(CRASHPAD_ZLIB_SOURCE_SYSTEM) || \
    defined(CRASHPAD_ZLIB_SOURCE_EXTERNAL)
#include <zlib.h>
#elif defined(CRASHPAD_ZLIB_SOURCE_EMBEDDED)
#include "third_party/zlib/zlib/zlib.h"
#else
#error Unknown zlib source
#endif

#endif  // CRASHPAD_THIRD_PARTY_ZLIB_ZLIB_CRASHPAD_H_
