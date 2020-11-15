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

#include "snapshot/unloaded_module_snapshot.h"

namespace crashpad {

UnloadedModuleSnapshot::UnloadedModuleSnapshot(uint64_t address,
                                               uint64_t size,
                                               uint32_t checksum,
                                               uint32_t timestamp,
                                               const std::string& name)
    : name_(name),
      address_(address),
      size_(size),
      checksum_(checksum),
      timestamp_(timestamp) {}

UnloadedModuleSnapshot::~UnloadedModuleSnapshot() {
}

}  // namespace crashpad
