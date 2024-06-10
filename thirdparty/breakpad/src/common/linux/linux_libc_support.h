// Copyright 2009 Google LLC
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
//     * Neither the name of Google LLC nor the names of its
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

// This header provides replacements for libc functions that we need. We if
// call the libc functions directly we risk crashing in the dynamic linker as
// it tries to resolve uncached PLT entries.

#ifndef CLIENT_LINUX_LINUX_LIBC_SUPPORT_H_
#define CLIENT_LINUX_LINUX_LIBC_SUPPORT_H_

#include <stdint.h>
#include <limits.h>
#include <sys/types.h>

extern "C" {

extern size_t my_strlen(const char* s);

extern int my_strcmp(const char* a, const char* b);

extern int my_strncmp(const char* a, const char* b, size_t len);

// Parse a non-negative integer.
//   result: (output) the resulting non-negative integer
//   s: a NUL terminated string
// Return true iff successful.
extern bool my_strtoui(int* result, const char* s);

// Return the length of the given unsigned integer when expressed in base 10.
extern unsigned my_uint_len(uintmax_t i);

// Convert an unsigned integer to a string
//   output: (output) the resulting string is written here. This buffer must be
//     large enough to hold the resulting string. Call |my_uint_len| to get the
//     required length.
//   i: the unsigned integer to serialise.
//   i_len: the length of the integer in base 10 (see |my_uint_len|).
extern void my_uitos(char* output, uintmax_t i, unsigned i_len);

extern const char* my_strchr(const char* haystack, char needle);

extern const char* my_strrchr(const char* haystack, char needle);

// Read a hex value
//   result: (output) the resulting value
//   s: a string
// Returns a pointer to the first invalid charactor.
extern const char* my_read_hex_ptr(uintptr_t* result, const char* s);

extern const char* my_read_decimal_ptr(uintptr_t* result, const char* s);

extern void my_memset(void* ip, char c, size_t len);

extern void* my_memchr(const void* src, int c, size_t len);

// The following are considered safe to use in a compromised environment.
// Besides, this gives the compiler an opportunity to optimize their calls.
#define my_memcpy  memcpy
#define my_memmove memmove
#define my_memcmp  memcmp

extern size_t my_strlcpy(char* s1, const char* s2, size_t len);

extern size_t my_strlcat(char* s1, const char* s2, size_t len);

extern int my_isspace(int ch);

}  // extern "C"

#endif  // CLIENT_LINUX_LINUX_LIBC_SUPPORT_H_
