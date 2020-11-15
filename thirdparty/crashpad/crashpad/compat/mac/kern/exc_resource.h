// Copyright 2015 The Crashpad Authors. All rights reserved.
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

#ifndef CRASHPAD_COMPAT_MAC_KERN_EXC_RESOURCE_H_
#define CRASHPAD_COMPAT_MAC_KERN_EXC_RESOURCE_H_

#if __has_include_next(<kern/exc_resource.h>)
#include_next <kern/exc_resource.h>
#endif

// 10.9 SDK

#ifndef EXC_RESOURCE_DECODE_RESOURCE_TYPE
#define EXC_RESOURCE_DECODE_RESOURCE_TYPE(code) (((code) >> 61) & 0x7ull)
#endif

#ifndef EXC_RESOURCE_DECODE_FLAVOR
#define EXC_RESOURCE_DECODE_FLAVOR(code) (((code) >> 58) & 0x7ull)
#endif

#ifndef RESOURCE_TYPE_CPU
#define RESOURCE_TYPE_CPU 1
#endif

#ifndef RESOURCE_TYPE_WAKEUPS
#define RESOURCE_TYPE_WAKEUPS 2
#endif

#ifndef RESOURCE_TYPE_MEMORY
#define RESOURCE_TYPE_MEMORY 3
#endif

#ifndef FLAVOR_CPU_MONITOR
#define FLAVOR_CPU_MONITOR 1
#endif

#ifndef FLAVOR_WAKEUPS_MONITOR
#define FLAVOR_WAKEUPS_MONITOR 1
#endif

#ifndef FLAVOR_HIGH_WATERMARK
#define FLAVOR_HIGH_WATERMARK 1
#endif

// 10.10 SDK

#ifndef FLAVOR_CPU_MONITOR_FATAL
#define FLAVOR_CPU_MONITOR_FATAL 2
#endif

// 10.12 SDK

#ifndef RESOURCE_TYPE_IO
#define RESOURCE_TYPE_IO 4
#endif

#ifndef FLAVOR_IO_PHYSICAL_WRITES
#define FLAVOR_IO_PHYSICAL_WRITES 1
#endif

#ifndef FLAVOR_IO_LOGICAL_WRITES
#define FLAVOR_IO_LOGICAL_WRITES 2
#endif

#endif  // CRASHPAD_COMPAT_MAC_KERN_EXC_RESOURCE_H_
