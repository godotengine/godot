// Copyright (c) 2006, Google Inc.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// macho_utilities.h: Utilities for dealing with mach-o files
//
// Author: Dave Camp

#ifndef COMMON_MAC_MACHO_UTILITIES_H__
#define COMMON_MAC_MACHO_UTILITIES_H__

#include <mach-o/loader.h>
#include <mach/thread_status.h>

/* Some #defines and structs that aren't defined in older SDKs */
#ifndef CPU_ARCH_ABI64
# define CPU_ARCH_ABI64    0x01000000
#endif

#ifndef CPU_TYPE_X86
# define CPU_TYPE_X86 CPU_TYPE_I386
#endif

#ifndef CPU_TYPE_POWERPC64
# define CPU_TYPE_POWERPC64 (CPU_TYPE_POWERPC | CPU_ARCH_ABI64)
#endif

#ifndef LC_UUID
# define LC_UUID         0x1b    /* the uuid */
#endif

// The uuid_command struct/swap routines were added during the 10.4 series.
// Their presence isn't guaranteed.
struct breakpad_uuid_command {
  uint32_t    cmd;            /* LC_UUID */
  uint32_t    cmdsize;        /* sizeof(struct uuid_command) */
  uint8_t     uuid[16];       /* the 128-bit uuid */
};

void breakpad_swap_uuid_command(struct breakpad_uuid_command *uc);

void breakpad_swap_load_command(struct load_command *lc);

void breakpad_swap_dylib_command(struct dylib_command *dc);

// Older SDKs defines thread_state_data_t as an int[] instead
// of the natural_t[] it should be.
typedef natural_t breakpad_thread_state_data_t[THREAD_STATE_MAX];

void breakpad_swap_segment_command(struct segment_command *sc);

// The 64-bit swap routines were added during the 10.4 series, their
// presence isn't guaranteed.
void breakpad_swap_segment_command_64(struct segment_command_64 *sg);

void breakpad_swap_fat_header(struct fat_header *fh);

void breakpad_swap_fat_arch(struct fat_arch *fa, uint32_t narchs);

void breakpad_swap_mach_header(struct mach_header *mh);

void breakpad_swap_mach_header_64(struct mach_header_64 *mh);

void breakpad_swap_section(struct section *s,
                           uint32_t nsects);

void breakpad_swap_section_64(struct section_64 *s,
                              uint32_t nsects);

#endif
