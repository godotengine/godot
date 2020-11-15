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

#include "util/posix/scoped_mmap.h"

#include <unistd.h>

#include <algorithm>

#include "base/logging.h"
#include "base/numerics/safe_math.h"

namespace {

bool Munmap(uintptr_t addr, size_t len) {
  if (munmap(reinterpret_cast<void*>(addr), len) != 0) {
    PLOG(ERROR) << "munmap";
    return false;
  }

  return true;
}

}  // namespace

namespace crashpad {

ScopedMmap::ScopedMmap() {}

ScopedMmap::~ScopedMmap() {
  if (is_valid()) {
    Munmap(reinterpret_cast<uintptr_t>(addr_), len_);
  }
}

bool ScopedMmap::Reset() {
  return ResetAddrLen(MAP_FAILED, 0);
}

bool ScopedMmap::ResetAddrLen(void* addr, size_t len) {
  const uintptr_t new_addr = reinterpret_cast<uintptr_t>(addr);

  if (addr == MAP_FAILED) {
    DCHECK_EQ(len, 0u);
  } else {
    DCHECK_NE(len, 0u);
    DCHECK_EQ(new_addr % getpagesize(), 0u);
    DCHECK_EQ(len % getpagesize(), 0u);
    DCHECK((base::CheckedNumeric<uintptr_t>(new_addr) + (len - 1)).IsValid());
  }

  bool result = true;

  if (is_valid()) {
    const uintptr_t old_addr = reinterpret_cast<uintptr_t>(addr_);
    if (old_addr < new_addr) {
      result &= Munmap(old_addr, std::min(len_, new_addr - old_addr));
    }
    if (old_addr + len_ > new_addr + len) {
      uintptr_t unmap_start = std::max(old_addr, new_addr + len);
      result &= Munmap(unmap_start, old_addr + len_ - unmap_start);
    }
  }

  addr_ = addr;
  len_ = len;

  return result;
}

bool ScopedMmap::ResetMmap(void* addr,
                           size_t len,
                           int prot,
                           int flags,
                           int fd,
                           off_t offset) {
  // Reset() first, so that a new anonymous mapping can use the address space
  // occupied by the old mapping if appropriate. The new mapping will be
  // attempted even if there was something wrong with the old mapping, so don’t
  // consider the return value from Reset().
  Reset();

  void* new_addr = mmap(addr, len, prot, flags, fd, offset);
  if (new_addr == MAP_FAILED) {
    PLOG(ERROR) << "mmap";
    return false;
  }

  // The new mapping is effective even if there was something wrong with the old
  // mapping, so don’t consider the return value from ResetAddrLen().
  ResetAddrLen(new_addr, len);
  return true;
}

bool ScopedMmap::Mprotect(int prot) {
  if (mprotect(addr_, len_, prot) < 0) {
    PLOG(ERROR) << "mprotect";
    return false;
  }

  return true;
}

}  // namespace crashpad
