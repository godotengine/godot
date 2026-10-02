/*
 MIT License

 Copyright (c) 2025 Catena cyber

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.
*/

#ifndef NALLOC_H_
#define NALLOC_H_

#if defined(__clang__) && defined(__has_feature)
#if __has_feature(address_sanitizer)
#define NALLOC_ASAN 1
#endif
#if __has_feature(memory_sanitizer)
#define FUZZER_DISABLE_NALLOC 1
#endif
#endif

#if defined(FUZZER_DISABLE_NALLOC)
#define nalloc_init(x)
#define nalloc_restrict_file_prefix(x)
#define nalloc_start(x, y)
#define nalloc_end()
#else

#include <errno.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

static const uint32_t nalloc_crc32_table[] = {
  0x00000000, 0x04c11db7, 0x09823b6e, 0x0d4326d9, 0x130476dc, 0x17c56b6b,
  0x1a864db2, 0x1e475005, 0x2608edb8, 0x22c9f00f, 0x2f8ad6d6, 0x2b4bcb61,
  0x350c9b64, 0x31cd86d3, 0x3c8ea00a, 0x384fbdbd, 0x4c11db70, 0x48d0c6c7,
  0x4593e01e, 0x4152fda9, 0x5f15adac, 0x5bd4b01b, 0x569796c2, 0x52568b75,
  0x6a1936c8, 0x6ed82b7f, 0x639b0da6, 0x675a1011, 0x791d4014, 0x7ddc5da3,
  0x709f7b7a, 0x745e66cd, 0x9823b6e0, 0x9ce2ab57, 0x91a18d8e, 0x95609039,
  0x8b27c03c, 0x8fe6dd8b, 0x82a5fb52, 0x8664e6e5, 0xbe2b5b58, 0xbaea46ef,
  0xb7a96036, 0xb3687d81, 0xad2f2d84, 0xa9ee3033, 0xa4ad16ea, 0xa06c0b5d,
  0xd4326d90, 0xd0f37027, 0xddb056fe, 0xd9714b49, 0xc7361b4c, 0xc3f706fb,
  0xceb42022, 0xca753d95, 0xf23a8028, 0xf6fb9d9f, 0xfbb8bb46, 0xff79a6f1,
  0xe13ef6f4, 0xe5ffeb43, 0xe8bccd9a, 0xec7dd02d, 0x34867077, 0x30476dc0,
  0x3d044b19, 0x39c556ae, 0x278206ab, 0x23431b1c, 0x2e003dc5, 0x2ac12072,
  0x128e9dcf, 0x164f8078, 0x1b0ca6a1, 0x1fcdbb16, 0x018aeb13, 0x054bf6a4,
  0x0808d07d, 0x0cc9cdca, 0x7897ab07, 0x7c56b6b0, 0x71159069, 0x75d48dde,
  0x6b93dddb, 0x6f52c06c, 0x6211e6b5, 0x66d0fb02, 0x5e9f46bf, 0x5a5e5b08,
  0x571d7dd1, 0x53dc6066, 0x4d9b3063, 0x495a2dd4, 0x44190b0d, 0x40d816ba,
  0xaca5c697, 0xa864db20, 0xa527fdf9, 0xa1e6e04e, 0xbfa1b04b, 0xbb60adfc,
  0xb6238b25, 0xb2e29692, 0x8aad2b2f, 0x8e6c3698, 0x832f1041, 0x87ee0df6,
  0x99a95df3, 0x9d684044, 0x902b669d, 0x94ea7b2a, 0xe0b41de7, 0xe4750050,
  0xe9362689, 0xedf73b3e, 0xf3b06b3b, 0xf771768c, 0xfa325055, 0xfef34de2,
  0xc6bcf05f, 0xc27dede8, 0xcf3ecb31, 0xcbffd686, 0xd5b88683, 0xd1799b34,
  0xdc3abded, 0xd8fba05a, 0x690ce0ee, 0x6dcdfd59, 0x608edb80, 0x644fc637,
  0x7a089632, 0x7ec98b85, 0x738aad5c, 0x774bb0eb, 0x4f040d56, 0x4bc510e1,
  0x46863638, 0x42472b8f, 0x5c007b8a, 0x58c1663d, 0x558240e4, 0x51435d53,
  0x251d3b9e, 0x21dc2629, 0x2c9f00f0, 0x285e1d47, 0x36194d42, 0x32d850f5,
  0x3f9b762c, 0x3b5a6b9b, 0x0315d626, 0x07d4cb91, 0x0a97ed48, 0x0e56f0ff,
  0x1011a0fa, 0x14d0bd4d, 0x19939b94, 0x1d528623, 0xf12f560e, 0xf5ee4bb9,
  0xf8ad6d60, 0xfc6c70d7, 0xe22b20d2, 0xe6ea3d65, 0xeba91bbc, 0xef68060b,
  0xd727bbb6, 0xd3e6a601, 0xdea580d8, 0xda649d6f, 0xc423cd6a, 0xc0e2d0dd,
  0xcda1f604, 0xc960ebb3, 0xbd3e8d7e, 0xb9ff90c9, 0xb4bcb610, 0xb07daba7,
  0xae3afba2, 0xaafbe615, 0xa7b8c0cc, 0xa379dd7b, 0x9b3660c6, 0x9ff77d71,
  0x92b45ba8, 0x9675461f, 0x8832161a, 0x8cf30bad, 0x81b02d74, 0x857130c3,
  0x5d8a9099, 0x594b8d2e, 0x5408abf7, 0x50c9b640, 0x4e8ee645, 0x4a4ffbf2,
  0x470cdd2b, 0x43cdc09c, 0x7b827d21, 0x7f436096, 0x7200464f, 0x76c15bf8,
  0x68860bfd, 0x6c47164a, 0x61043093, 0x65c52d24, 0x119b4be9, 0x155a565e,
  0x18197087, 0x1cd86d30, 0x029f3d35, 0x065e2082, 0x0b1d065b, 0x0fdc1bec,
  0x3793a651, 0x3352bbe6, 0x3e119d3f, 0x3ad08088, 0x2497d08d, 0x2056cd3a,
  0x2d15ebe3, 0x29d4f654, 0xc5a92679, 0xc1683bce, 0xcc2b1d17, 0xc8ea00a0,
  0xd6ad50a5, 0xd26c4d12, 0xdf2f6bcb, 0xdbee767c, 0xe3a1cbc1, 0xe760d676,
  0xea23f0af, 0xeee2ed18, 0xf0a5bd1d, 0xf464a0aa, 0xf9278673, 0xfde69bc4,
  0x89b8fd09, 0x8d79e0be, 0x803ac667, 0x84fbdbd0, 0x9abc8bd5, 0x9e7d9662,
  0x933eb0bb, 0x97ffad0c, 0xafb010b1, 0xab710d06, 0xa6322bdf, 0xa2f33668,
  0xbcb4666d, 0xb8757bda, 0xb5365d03, 0xb1f740b4
};

// Nallocfuzz data to take a decision
uint32_t nalloc_random_state = 0;
__thread unsigned int nalloc_running = 0;
bool nalloc_initialized = false;
uint32_t nalloc_runs = 0;

// Nalloc fuzz parameters
uint32_t nalloc_bitmask = 0xFF;
bool nalloc_random_bitmask = true;
uint32_t nalloc_magic = 0x294cee63;
bool nalloc_verbose = false;

#ifdef NALLOC_ASAN
extern void __sanitizer_print_stack_trace(void);
#endif

// Generic init, using env variables to get parameters
void nalloc_init(const char *prog) {
  if (nalloc_initialized) {
    return;
  }
  nalloc_initialized = true;
  char *bitmask = getenv("NALLOC_FREQ");
  if (bitmask) {
    int shift = atoi(bitmask);
    if (shift > 0 && shift < 31) {
      nalloc_bitmask = 1 << shift;
      nalloc_random_bitmask = false;
    } else if (shift == 0) {
      nalloc_random_bitmask = false;
      nalloc_bitmask = 0;
    }
  } else if (prog == NULL || strstr(prog, "nalloc") == NULL) {
    nalloc_random_bitmask = false;
    nalloc_bitmask = 0;
    return;
  }

  char *magic = getenv("NALLOC_MAGIC");
  if (magic) {
    nalloc_magic = (uint32_t)strtol(magic, NULL, 0);
  }

  char *verbose = getenv("NALLOC_VERBOSE");
  if (verbose) {
    nalloc_verbose = true;
  }
}

// add one byte to the CRC
static inline void nalloc_random_update(uint8_t b) {
  nalloc_random_state =
      ((uint32_t)((uint32_t)nalloc_random_state << 8)) ^
      nalloc_crc32_table[((nalloc_random_state >> 24) ^ b) & 0xFF];
}

// Start the failure injections, using a buffer as seed
static int nalloc_start(const uint8_t *data, size_t size) {
  if (nalloc_random_bitmask) {
    if (nalloc_random_state & 0x10) {
      nalloc_bitmask = 0xFFFFFFFF;
    } else {
      nalloc_bitmask = 1 << (5 + (nalloc_random_state & 0xF));
    }
  } else if (nalloc_bitmask == 0) {
    // nalloc disabled
    return 0;
  }
  nalloc_random_state = 0;
  for (size_t i = 0; i < size; i++) {
    nalloc_random_update(data[i]);
  }
  if (__sync_fetch_and_add(&nalloc_running, 1)) {
    __sync_fetch_and_sub(&nalloc_running, 1);
    return 0;
  }
  nalloc_runs++;
  return 1;
}

// Stop the failure injections
static void nalloc_end() { __sync_fetch_and_sub(&nalloc_running, 1); }

static bool nalloc_backtrace_exclude(size_t size, const char *op) {
  if (nalloc_verbose) {
    fprintf(stderr, "failed %s(%zu) \n", op, size);
#ifdef NALLOC_ASAN
    __sanitizer_print_stack_trace();
#endif
  }

  return false;
}

//
static bool nalloc_fail(size_t size, const char *op) {
  // do not fail before thread init
  if (nalloc_runs == 0) {
    return false;
  }
  if (__sync_fetch_and_add(&nalloc_running, 1) != 1) {
    // do not fail allocations outside of fuzzer input
    // and do not fail inside of this function
    __sync_fetch_and_sub(&nalloc_running, 1);
    return false;
  }
  nalloc_random_update((uint8_t)size);
  if (size >= 0x100) {
    nalloc_random_update((uint8_t)(size >> 8));
    if (size >= 0x10000) {
      nalloc_random_update((uint8_t)(size >> 16));
      // bigger may already fail or oom
    }
  }
  if (((nalloc_random_state ^ nalloc_magic) & nalloc_bitmask) == 0) {
    if (nalloc_backtrace_exclude(size, op)) {
      __sync_fetch_and_sub(&nalloc_running, 1);
      return false;
    }
    __sync_fetch_and_sub(&nalloc_running, 1);
    return true;
  }
  __sync_fetch_and_sub(&nalloc_running, 1);
  return false;
}

// ASAN interceptor for libc routines
#ifdef NALLOC_ASAN
extern void *__interceptor_malloc(size_t);
extern void *__interceptor_calloc(size_t, size_t);
extern void *__interceptor_realloc(void *, size_t);
extern void *__interceptor_reallocarray(void *, size_t, size_t);

extern ssize_t __interceptor_read(int, void *, size_t);
extern ssize_t __interceptor_write(int, const void *, size_t);
extern ssize_t __interceptor_recv(int, void *, size_t, int);
extern ssize_t __interceptor_send(int, const void *, size_t, int);

#define nalloc_malloc(s) __interceptor_malloc(s)
#define nalloc_calloc(s, n) __interceptor_calloc(s, n)
#define nalloc_realloc(p, s) __interceptor_realloc(p, s)
#define nalloc_reallocarray(p, s, n) __interceptor_reallocarray(p, s, n)

#define nalloc_read(f, b, s) __interceptor_read(f, b, s)
#define nalloc_write(f, b, s) __interceptor_write(f, b, s)
#define nalloc_recv(f, b, s, x) __interceptor_recv(f, b, s, x)
#define nalloc_send(f, b, s, x) __interceptor_send(f, b, s, x)

#else
extern void *__libc_malloc(size_t);
extern void *__libc_calloc(size_t, size_t);
extern void *__libc_realloc(void *, size_t);
extern void *__libc_reallocarray(void *, size_t, size_t);

extern ssize_t __read(int, void *, size_t);
extern ssize_t __write(int, const void *, size_t);
extern ssize_t __recv(int, void *, size_t, int);
extern ssize_t __send(int, const void *, size_t, int);

#define nalloc_malloc(s) __libc_malloc(s)
#define nalloc_calloc(s, n) __libc_calloc(s, n)
#define nalloc_realloc(p, s) __libc_realloc(p, s)
#define nalloc_reallocarray(p, s, n) __libc_reallocarray(p, s, n)

#define nalloc_read(f, b, s) __read(f, b, s)
#define nalloc_write(f, b, s) __write(f, b, s)
#define nalloc_recv(f, b, s, x) __recv(f, b, s, x)
#define nalloc_send(f, b, s, x) __send(f, b, s, x)
#endif

// nalloc standard function overwrites with pseudo-random failures
ssize_t read(int fd, void *buf, size_t count) {
  if (nalloc_fail(count, "read")) {
    errno = EIO;
    return -1;
  }
  return nalloc_read(fd, buf, count);
}

ssize_t write(int fd, const void *buf, size_t count) {
  if (nalloc_fail(count, "write")) {
    errno = EIO;
    return -1;
  }
  return nalloc_write(fd, buf, count);
}

ssize_t recv(int fd, void *buf, size_t count, int flags) {
  if (nalloc_fail(count, "recv")) {
    errno = EIO;
    return -1;
  }
  return nalloc_recv(fd, buf, count, flags);
}

ssize_t send(int fd, const void *buf, size_t count, int flags) {
  if (nalloc_fail(count, "send")) {
    errno = EIO;
    return -1;
  }
  return nalloc_send(fd, buf, count, flags);
}

void *calloc(size_t nmemb, size_t size) {
  if (nalloc_fail(size, "calloc")) {
    errno = ENOMEM;
    return NULL;
  }
  return nalloc_calloc(nmemb, size);
}

void *malloc(size_t size) {
  if (nalloc_fail(size, "malloc")) {
    errno = ENOMEM;
    return NULL;
  }
  return nalloc_malloc(size);
}

void *realloc(void *ptr, size_t size) {
  if (nalloc_fail(size, "realloc")) {
    errno = ENOMEM;
    return NULL;
  }
  return nalloc_realloc(ptr, size);
}

void *reallocarray(void *ptr, size_t nmemb, size_t size) {
  if (nalloc_fail(size, "reallocarray")) {
    errno = ENOMEM;
    return NULL;
  }
  return nalloc_reallocarray(ptr, nmemb, size);
}

#ifdef __cplusplus
}  // extern "C" {
#endif

#endif // FUZZER_DISABLE_NALLOC

#endif  // NALLOC_H_
