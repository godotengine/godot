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

#include <sys/mman.h>

#include <dlfcn.h>
#include <errno.h>
#include <stdint.h>
#include <unistd.h>

#include "dlfcn_internal.h"

#if defined(__USE_FILE_OFFSET64) && __ANDROID_API__ < 21

// Bionic has provided a wrapper for __mmap2() since the beginning of time. See
// bionic/libc/SYSCALLS.TXT in any Android version.
extern "C" void* __mmap2(void* addr,
                         size_t size,
                         int prot,
                         int flags,
                         int fd,
                         size_t pgoff);

namespace {

template <typename T>
T Align(T value, uint8_t alignment) {
  return (value + alignment - 1) & ~(alignment - 1);
}

// Adapted from Android 8.0.0 bionic/libc/bionic/mmap.cpp.
void* LocalMmap64(void* addr,
                  size_t size,
                  int prot,
                  int flags,
                  int fd,
                  off64_t offset) {
  constexpr int kMmap2Shift = 12;

  if (offset < 0 || (offset & ((1UL << kMmap2Shift) - 1)) != 0) {
    errno = EINVAL;
    return MAP_FAILED;
  }

  const size_t rounded = Align(size, getpagesize());
  if (rounded < size || rounded > PTRDIFF_MAX) {
    errno = ENOMEM;
    return MAP_FAILED;
  }

  const bool is_private_anonymous =
      (flags & (MAP_PRIVATE | MAP_ANONYMOUS)) == (MAP_PRIVATE | MAP_ANONYMOUS);
  const bool is_stack_or_grows_down =
      (flags & (MAP_STACK | MAP_GROWSDOWN)) != 0;

  void* const result =
      __mmap2(addr, size, prot, flags, fd, offset >> kMmap2Shift);

  static bool kernel_has_MADV_MERGEABLE = true;
  if (result != MAP_FAILED && kernel_has_MADV_MERGEABLE &&
      is_private_anonymous && !is_stack_or_grows_down) {
    const int saved_errno = errno;
    const int rc = madvise(result, size, MADV_MERGEABLE);
    if (rc == -1 && errno == EINVAL) {
      kernel_has_MADV_MERGEABLE = false;
    }
    errno = saved_errno;
  }

  return result;
}

}  // namespace

extern "C" {

void* mmap(void* addr, size_t size, int prot, int flags, int fd, off_t offset) {
  // Use the system’s mmap64() wrapper if available. It will be available on
  // Android 5.0 (“Lollipop”) and later.
  using Mmap64Type = void* (*)(void*, size_t, int, int, int, off64_t);
  static const Mmap64Type mmap64 = reinterpret_cast<Mmap64Type>(
      crashpad::internal::Dlsym(RTLD_DEFAULT, "mmap64"));
  if (mmap64) {
    return mmap64(addr, size, prot, flags, fd, offset);
  }

  // Otherwise, use the local implementation, which should amount to exactly the
  // same thing.
  return LocalMmap64(addr, size, prot, flags, fd, offset);
}

}  // extern "C"

#endif  // defined(__USE_FILE_OFFSET64) && __ANDROID_API__ < 21
