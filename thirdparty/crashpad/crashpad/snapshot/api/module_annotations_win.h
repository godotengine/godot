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

#ifndef CRASHPAD_SNAPSHOT_API_MODULE_ANNOTATIONS_WIN_H_
#define CRASHPAD_SNAPSHOT_API_MODULE_ANNOTATIONS_WIN_H_

#include <windows.h>

#include <map>
#include <string>

namespace crashpad {

//! \brief Reads the module annotations from another process.
//!
//! \param[in] process The handle to the process that hosts the \a module.
//!     Requires PROCESS_QUERY_INFORMATION and PROCESS_VM_READ accesses.
//! \param[in] module The handle to the module from which the \a annotations
//!     will be read. This module should be loaded in the target process.
//! \param[out] annotations The map that will be filled with the annotations.
//!     Remains unchanged if the function returns 'false'.
//!
//! \return `true` if the annotations could be read succesfully, even if the
//!     module doesn't contain any annotations.
bool ReadModuleAnnotations(HANDLE process,
                           HMODULE module,
                           std::map<std::string, std::string>* annotations);

}  // namespace crashpad

#endif  // CRASHPAD_SNAPSHOT_API_MODULE_ANNOTATIONS_WIN_H_
