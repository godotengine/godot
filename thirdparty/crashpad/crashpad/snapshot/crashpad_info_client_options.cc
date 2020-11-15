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

#include "snapshot/crashpad_info_client_options.h"

#include "base/logging.h"
#include "client/crashpad_info.h"

namespace crashpad {

// static
TriState CrashpadInfoClientOptions::TriStateFromCrashpadInfo(
    uint8_t crashpad_info_tri_state) {
  switch (crashpad_info_tri_state) {
    case static_cast<uint8_t>(TriState::kUnset):
      return TriState::kUnset;
    case static_cast<uint8_t>(TriState::kEnabled):
      return TriState::kEnabled;
    case static_cast<uint8_t>(TriState::kDisabled):
      return TriState::kDisabled;
    default:
      LOG(WARNING) << "unknown TriState "
                   << static_cast<int>(crashpad_info_tri_state);
      return TriState::kUnset;
  }
}

CrashpadInfoClientOptions::CrashpadInfoClientOptions()
    : crashpad_handler_behavior(TriState::kUnset),
      system_crash_reporter_forwarding(TriState::kUnset),
      gather_indirectly_referenced_memory(TriState::kUnset) {
}

}  // namespace crashpad
