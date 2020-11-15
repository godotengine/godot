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

#ifndef CRASHPAD_TEST_MAC_DYLD_H_
#define CRASHPAD_TEST_MAC_DYLD_H_

#include <mach-o/dyld_images.h>

namespace crashpad {
namespace test {

//! \brief Calls or emulates the `_dyld_get_all_image_infos()` private/internal
//!     function.
//!
//! \return A pointer to this process’ dyld_all_image_infos structure, or
//!     `nullptr` on failure with a message logged.
const dyld_all_image_infos* DyldGetAllImageInfos();

}  // namespace test
}  // namespace crashpad

#endif  // CRASHPAD_TEST_MAC_DYLD_H_
