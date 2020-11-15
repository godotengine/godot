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

#ifndef CRASHPAD_UTIL_POSIX_DROP_PRIVILEGES_H_
#define CRASHPAD_UTIL_POSIX_DROP_PRIVILEGES_H_

namespace crashpad {

//! \brief Permanently drops privileges conferred by being a setuid or setgid
//!     executable.
//!
//! The effective user ID and saved set-user ID are set to the real user ID,
//! negating any effects of being a setuid executable. The effective group ID
//! and saved set-group ID are set to the real group ID, negating any effects of
//! being a setgid executable. Because the saved set-user ID and saved set-group
//! ID are reset, there is no way to restore the prior privileges, and the drop
//! is permanent.
//!
//! This function drops privileges correctly when running setuid root and in
//! other circumstances, including when running setuid non-root. If the program
//! is not a setuid or setgid executable, this function has no effect.
//!
//! No changes are made to the supplementary group list, which is normally not
//! altered for setuid or setgid executables.
void DropPrivileges();

}  // namespace crashpad

#endif  // CRASHPAD_UTIL_POSIX_DROP_PRIVILEGES_H_
