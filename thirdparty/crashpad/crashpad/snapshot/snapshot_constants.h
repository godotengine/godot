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

#ifndef SNAPSHOT_SNAPSHOT_CONSTANTS_H_
#define SNAPSHOT_SNAPSHOT_CONSTANTS_H_

namespace crashpad {

//! \brief The maximum number of crashpad::Annotations that will be read from
//!     a client process.
//!
//! \note This maximum was chosen arbitrarily and may change in the future.
constexpr size_t kMaxNumberOfAnnotations = 200;

}  // namespace crashpad

#endif  // SNAPSHOT_SNAPSHOT_CONSTANTS_H_
