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

#ifndef CRASHPAD_COMPAT_ANDROID_ELF_H_
#define CRASHPAD_COMPAT_ANDROID_ELF_H_

#include_next <elf.h>

#include <android/api-level.h>

#if !defined(ELF32_ST_VISIBILITY)
#define ELF32_ST_VISIBILITY(other) ((other) & 0x3)
#endif

#if !defined(ELF64_ST_VISIBILITY)
#define ELF64_ST_VISIBILITY(other) ELF32_ST_VISIBILITY(other)
#endif

// Android 5.0.0 (API 21) NDK

#if !defined(STT_COMMON)
#define STT_COMMON 5
#endif

#if !defined(STT_TLS)
#define STT_TLS 6
#endif

// ELF note header types are normally provided by <linux/elf.h>. While unified
// headers include <linux/elf.h> in <elf.h>, traditional headers do not, prior
// to API 21. <elf.h> and <linux/elf.h> can't both be included in the same
// translation unit due to collisions, so we provide these types here.
#if __ANDROID_API__ < 21 && !defined(__ANDROID_API_N__)
typedef struct {
  Elf32_Word n_namesz;
  Elf32_Word n_descsz;
  Elf32_Word n_type;
} Elf32_Nhdr;

typedef struct {
  Elf64_Word n_namesz;
  Elf64_Word n_descsz;
  Elf64_Word n_type;
} Elf64_Nhdr;
#endif  // __ANDROID_API__ < 21 && !defined(NT_PRSTATUS)

#endif  // CRASHPAD_COMPAT_ANDROID_ELF_H_
