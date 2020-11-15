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

#ifndef CRASHPAD_SNAPSHOT_CRASHPAD_TYPES_CRASHPAD_INFO_READER_H_
#define CRASHPAD_SNAPSHOT_CRASHPAD_TYPES_CRASHPAD_INFO_READER_H_

#include <stdint.h>

#include <memory>

#include "base/macros.h"
#include "util/misc/address_types.h"
#include "util/misc/initialization_state_dcheck.h"
#include "util/misc/tri_state.h"
#include "util/process/process_memory_range.h"

namespace crashpad {

//! \brief Reads CrashpadInfo structs from another process via a
//!     ProcessMemoryRange.
class CrashpadInfoReader {
 public:
  CrashpadInfoReader();
  ~CrashpadInfoReader();

  //! \brief Initializes this object.
  //!
  //! This method must be successfully called bfore any other method in this
  //! class.
  //!
  //! \param[in] memory The reader for the remote process.
  //! \param[in] address The address in the remote process' address space of a
  //!     CrashpadInfo struct.
  //! \return `true` on success. `false` on failure with a message logged.
  bool Initialize(const ProcessMemoryRange* memory, VMAddress address);

  //! \{
  //! \see CrashpadInfo
  TriState CrashpadHandlerBehavior();
  TriState SystemCrashReporterForwarding();
  TriState GatherIndirectlyReferencedMemory();
  uint32_t IndirectlyReferencedMemoryCap();
  VMAddress ExtraMemoryRanges();
  VMAddress SimpleAnnotations();
  VMAddress AnnotationsList();
  VMAddress UserDataMinidumpStreamHead();
  //! \}

 private:
  class InfoContainer;

  template <typename Traits>
  class InfoContainerSpecific;

  std::unique_ptr<InfoContainer> container_;
  bool is_64_bit_;
  InitializationStateDcheck initialized_;

  DISALLOW_COPY_AND_ASSIGN(CrashpadInfoReader);
};

}  // namespace crashpad

#endif  // CRASHPAD_SNAPSHOT_CRASHPAD_TYPES_CRASHPAD_INFO_READER_H_
