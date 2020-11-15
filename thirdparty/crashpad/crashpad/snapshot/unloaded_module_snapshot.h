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

#ifndef CRASHPAD_SNAPSHOT_UNLOADED_MODULE_SNAPSHOT_H_
#define CRASHPAD_SNAPSHOT_UNLOADED_MODULE_SNAPSHOT_H_

#include <stdint.h>

#include <string>

namespace crashpad {

//! \brief Information about an unloaded module that was previously loaded into
//!     a snapshot process.
class UnloadedModuleSnapshot {
 public:
  UnloadedModuleSnapshot(uint64_t address,
                  uint64_t size,
                  uint32_t checksum,
                  uint32_t timestamp,
                  const std::string& name);
  ~UnloadedModuleSnapshot();

  //! \brief The base address of the module in the target processes' address
  //!     space.
  uint64_t Address() const { return address_; }

  //! \brief The size of the module.
  uint64_t Size() const { return size_; }

  //! \brief The checksum of the image.
  uint32_t Checksum() const { return checksum_; }

  //! \brief The time and date stamp in `time_t` format.
  uint32_t Timestamp() const { return timestamp_; }

  //! \brief The name of the module.
  std::string Name() const { return name_; }

 private:
  std::string name_;
  uint64_t address_;
  uint64_t size_;
  uint32_t checksum_;
  uint32_t timestamp_;
};

}  // namespace crashpad

#endif  // CRASHPAD_SNAPSHOT_UNLOADED_MODULE_SNAPSHOT_H_
