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

#ifndef CRASHPAD_COMPAT_NON_MAC_MACH_MACH_H_
#define CRASHPAD_COMPAT_NON_MAC_MACH_MACH_H_

//! \file

// <mach/exception_types.h>

//! \anchor EXC_x
//! \name EXC_*
//!
//! \brief Mach exception type definitions.
//! \{
#define EXC_BAD_ACCESS 1
#define EXC_BAD_INSTRUCTION 2
#define EXC_ARITHMETIC 3
#define EXC_EMULATION 4
#define EXC_SOFTWARE 5
#define EXC_BREAKPOINT 6
#define EXC_SYSCALL 7
#define EXC_MACH_SYSCALL 8
#define EXC_RPC_ALERT 9
#define EXC_CRASH 10
#define EXC_RESOURCE 11
#define EXC_GUARD 12
#define EXC_CORPSE_NOTIFY 13

#define EXC_TYPES_COUNT 14
//! \}

#endif  // CRASHPAD_COMPAT_NON_MAC_MACH_MACH_H_
