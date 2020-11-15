// Copyright 2015 The Crashpad Authors. All rights reserved.
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

#ifndef CRASHPAD_SNAPSHOT_CRASHPAD_INFO_CLIENT_OPTIONS_H_
#define CRASHPAD_SNAPSHOT_CRASHPAD_INFO_CLIENT_OPTIONS_H_

#include <stdint.h>

#include "util/misc/tri_state.h"

namespace crashpad {

//! \brief Options represented in a client’s CrashpadInfo structure.
//!
//! The CrashpadInfo structure is not suitable to expose client options in a
//! generic way at the snapshot level. This structure duplicates option-related
//! fields from the client structure for general use within the snapshot layer
//! and by users of this layer.
//!
//! For objects of this type corresponding to a module, option values are taken
//! from the module’s CrashpadInfo structure directly. If the module has no such
//! structure, option values appear unset.
//!
//! For objects of this type corresponding to an entire process, option values
//! are taken from the CrashpadInfo structures of modules within the process.
//! The first module found with a set value (enabled or disabled) will provide
//! an option value for the process. Different modules may provide values for
//! different options. If no module in the process sets a value for an option,
//! the option will appear unset for the process. If no module in the process
//! has a CrashpadInfo structure, all option values will appear unset.
struct CrashpadInfoClientOptions {
 public:
  //! \brief Converts `uint8_t` value to a TriState value.
  //!
  //! The process_types layer exposes TriState as a `uint8_t` rather than an
  //! enum type. This function converts these values into the equivalent enum
  //! values used in the snapshot layer.
  //!
  //! \return The TriState equivalent of \a crashpad_info_tri_state, if it is a
  //!     valid TriState value. Otherwise, logs a warning and returns
  //!     TriState::kUnset.
  static TriState TriStateFromCrashpadInfo(uint8_t crashpad_info_tri_state);

  CrashpadInfoClientOptions();

  //! \sa CrashpadInfo::set_crashpad_handler_behavior()
  TriState crashpad_handler_behavior;

  //! \sa CrashpadInfo::set_system_crash_reporter_forwarding()
  TriState system_crash_reporter_forwarding;

  //! \sa CrashpadInfo::set_gather_indirectly_referenced_memory()
  TriState gather_indirectly_referenced_memory;

  //! \sa CrashpadInfo::set_gather_indirectly_referenced_memory()
  uint32_t indirectly_referenced_memory_cap;
};

}  // namespace crashpad

#endif  // CRASHPAD_SNAPSHOT_CRASHPAD_INFO_CLIENT_OPTIONS_H_
