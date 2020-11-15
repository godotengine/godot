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

#ifndef CRASHPAD_SNAPSHOT_MAC_PROCESS_TYPES_INTERNAL_H_
#define CRASHPAD_SNAPSHOT_MAC_PROCESS_TYPES_INTERNAL_H_

#include "snapshot/mac/process_types.h"

// Declare Traits32 and Traits64, flavor-specific traits classes. These are
// private traits classes not for use outside of process type internals.
// TraitsGeneric is declared in snapshot/mac/process_types.h.

#include "snapshot/mac/process_types/traits.h"

#define PROCESS_TYPE_FLAVOR_TRAITS(lp_bits) \
  DECLARE_PROCESS_TYPE_TRAITS_CLASS(        \
      lp_bits, lp_bits, __attribute__((aligned(lp_bits / 8))))

#include "snapshot/mac/process_types/flavors.h"

#undef PROCESS_TYPE_FLAVOR_TRAITS

#undef DECLARE_PROCESS_TYPE_TRAITS_CLASS

#endif  // CRASHPAD_SNAPSHOT_MAC_PROCESS_TYPES_INTERNAL_H_
