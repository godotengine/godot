// Copyright 2014 The Crashpad Authors. All rights reserved.
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

#ifndef CRASHPAD_UTIL_MACH_CHILD_PORT_TYPES_H_
#define CRASHPAD_UTIL_MACH_CHILD_PORT_TYPES_H_

#include <mach/mach.h>
#include <stdint.h>

// This file is #included by C (non-C++) files, and must remain strictly C.

typedef mach_port_t child_port_server_t;
typedef uint64_t child_port_token_t;

#endif  // CRASHPAD_UTIL_MACH_CHILD_PORT_TYPES_H_
