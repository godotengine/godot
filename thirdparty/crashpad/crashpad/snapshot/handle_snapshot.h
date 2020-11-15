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

#ifndef CRASHPAD_SNAPSHOT_HANDLE_SNAPSHOT_H_
#define CRASHPAD_SNAPSHOT_HANDLE_SNAPSHOT_H_

#include <stdint.h>

#include <string>

namespace crashpad {

struct HandleSnapshot {
  HandleSnapshot();
  ~HandleSnapshot();

  //! \brief A UTF-8 string representation of the handle's type.
  std::string type_name;

  //! \brief The handle's value.
  uint32_t handle;

  //! \brief The attributes for the handle, e.g. `OBJ_INHERIT`,
  //!     `OBJ_CASE_INSENSITIVE`, etc.
  uint32_t attributes;

  //! \brief The ACCESS_MASK for the handle in this process.
  //!
  //! See
  //! https://blogs.msdn.microsoft.com/openspecification/2010/04/01/about-the-access_mask-structure/
  //! for more information.
  uint32_t granted_access;

  //! \brief The number of kernel references to the object that this handle
  //!     refers to.
  uint32_t pointer_count;

  //! \brief The number of open handles to the object that this handle refers
  //!     to.
  uint32_t handle_count;
};

}  // namespace crashpad

#endif  // CRASHPAD_SNAPSHOT_HANDLE_SNAPSHOT_H_
