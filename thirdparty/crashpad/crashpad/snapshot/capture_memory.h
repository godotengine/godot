// Copyright 2016 The Crashpad Authors. All rights reserved.
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

#ifndef CRASHPAD_SNAPSHOT_CAPTURE_MEMORY_H_
#define CRASHPAD_SNAPSHOT_CAPTURE_MEMORY_H_

#include <stdint.h>

#include <vector>

#include "snapshot/cpu_context.h"
#include "util/numeric/checked_range.h"

namespace crashpad {

class MemorySnapshot;

namespace internal {

class CaptureMemory {
 public:
  //! \brief An interface to a platform-specific process reader.
  class Delegate {
   public:
    virtual ~Delegate() {}

    //! \return `true` if the target process is a 64-bit process.
    virtual bool Is64Bit() const = 0;

    //! \brief Attempts to read \a num_bytes bytes from the target process
    //!     starting at address \a at into \a into.
    //!
    //! \return `true` if the entire region could be read, or `false` with an
    //!     error logged.
    virtual bool ReadMemory(uint64_t at,
                            uint64_t num_bytes,
                            void* into) const = 0;

    //! \brief Given a range to be read from the target process, returns a
    //! vector
    //!     of ranges, representing the readable portions of the original range.
    //!
    //! \param[in] range The range being identified.
    //!
    //! \return A vector of ranges corresponding to the portion of \a range that
    //!     is readable.
    virtual std::vector<CheckedRange<uint64_t>> GetReadableRanges(
        const CheckedRange<uint64_t, uint64_t>& range) const = 0;

    //! \brief Adds the given range representing a memory snapshot in the target
    //!     process to the result.
    virtual void AddNewMemorySnapshot(
        const CheckedRange<uint64_t, uint64_t>& range) = 0;
  };

  //! \brief For all registers that appear to be pointer-like in \a context,
  //!     captures a small amount of memory near their pointed to location.
  //!
  //! "Pointer-like" in this context means not too close to zero (signed or
  //! unsigned) so that there's a reasonable chance that the value is a pointer.
  //!
  //! \param[in] context The context to inspect.
  //! \param[in] delegate A Delegate that handles reading from the target
  //!     process and adding new ranges.
  static void PointedToByContext(const CPUContext& context, Delegate* delegate);

  //! \brief For all pointer-like values in a memory range of the target
  //! process,
  //!     captures a small amount of memory near the pointed to location.
  //!
  //! \param[in] memory An existing MemorySnapshot of the range to search. The
  //!     base address and size must be pointer-aligned and an integral number
  //!     of
  //!     pointers long.
  //! \param[in] delegate A Delegate that handles reading from the target
  //!     process and adding new ranges.
  static void PointedToByMemoryRange(const MemorySnapshot& memory,
                                     Delegate* delegate);

 private:
  DISALLOW_IMPLICIT_CONSTRUCTORS(CaptureMemory);
};

}  // namespace internal
}  // namespace crashpad

#endif  // CRASHPAD_SNAPSHOT_CAPTURE_MEMORY_H_
