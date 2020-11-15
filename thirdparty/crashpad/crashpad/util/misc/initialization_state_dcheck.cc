// Copyright 2014 The Crashpad Authors. All rights reserved.
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

#include "util/misc/initialization_state_dcheck.h"

namespace crashpad {

#if DCHECK_IS_ON()

InitializationStateDcheck::State InitializationStateDcheck::SetInitializing() {
  State old_state = state();
  if (old_state == kStateUninitialized) {
    set_invalid();
  }

  return old_state;
}

InitializationStateDcheck::State InitializationStateDcheck::SetValid() {
  State old_state = state();
  if (old_state == kStateInvalid) {
    set_valid();
  }

  return old_state;
}

#endif

}  // namespace crashpad
