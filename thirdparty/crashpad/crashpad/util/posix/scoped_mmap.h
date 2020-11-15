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

#ifndef CRASHPAD_UTIL_POSIX_SCOPED_MMAP_H_
#define CRASHPAD_UTIL_POSIX_SCOPED_MMAP_H_

#include "base/macros.h"

#include <sys/mman.h>
#include <sys/types.h>

#include "util/misc/from_pointer_cast.h"

namespace crashpad {

//! \brief Maintains a memory-mapped region created by `mmap()`.
//!
//! On destruction, any memory-mapped region managed by an object of this class
//! will be released by calling `munmap()`.
class ScopedMmap {
 public:
  ScopedMmap();
  ~ScopedMmap();

  //! \brief Releases the memory-mapped region by calling `munmap()`.
  //!
  //! \return `true` on success. `false` on failure, with a message logged.
  bool Reset();

  //! \brief Releases any existing memory-mapped region and sets the object to
  //!     maintain an already-established mapping.
  //!
  //! If \a addr and \a len indicate a region that overlaps with the existing
  //! memory-mapped region, only the portion of the existing memory-mapped
  //! region that does not overlap the new region, if any, will be released.
  //!
  //! \param[in] addr The base address of the existing memory-mapped region to
  //!     maintain.
  //! \param[in] len The size of the existing memory-mapped region to maintain.
  //!
  //! \return `true` on success. `false` on failure, with a message logged.
  bool ResetAddrLen(void* addr, size_t len);

  //! \brief Releases any existing memory-mapped region and establishes a new
  //!     one by calling `mmap()`.
  //!
  //! The parameters to this method are passed directly to `mmap()`.
  //!
  //! \return `true` on success. `false` on failure, with a message logged. A
  //!     message will also be logged on failure to release any existing
  //!     memory-mapped region, but this will not preclude `mmap()` from being
  //!     called or a new mapping from being established, and if such a call to
  //!     `mmap()` is successful, this method will return `true`.
  bool ResetMmap(void* addr,
                 size_t len,
                 int prot,
                 int flags,
                 int fd,
                 off_t offset);

  //! \brief Sets the protection of the memory-mapped region by calling
  //!     `mprotect()`.
  //!
  //! \a prot is passed directly to `mprotect()`.
  //!
  //! \return `true` on success. `false` on failure, with a message logged.
  bool Mprotect(int prot);

  //! \return Whether this object is managing a valid memory-mapped region.
  bool is_valid() const { return addr_ != MAP_FAILED; }

  //! \brief Returns the base address of the memory-mapped region.
  void* addr() const { return addr_; }

  //! \brief Returns the base address of the memory-mapped region, casted to
  //!     a type of the caller’s choosing.
  template <typename T>
  T addr_as() const {
    return FromPointerCast<T>(addr_);
  }

  //! \brief Returns the size of the memory-mapped region.
  size_t len() const { return len_; }

 private:
  void* addr_ = MAP_FAILED;
  size_t len_ = 0;

  DISALLOW_COPY_AND_ASSIGN(ScopedMmap);
};

}  // namespace crashpad

#endif  // CRASHPAD_UTIL_POSIX_SCOPED_MMAP_H_
