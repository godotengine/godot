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

#ifndef CRASHPAD_UTIL_POSIX_CLOSE_STDIO_H_
#define CRASHPAD_UTIL_POSIX_CLOSE_STDIO_H_

namespace crashpad {

//! \brief Closes `stdin` and `stdout` by opening `/dev/null` over them.
//!
//! It is normally inadvisable to `close()` the three standard input/output
//! streams, because they occupy special file descriptors. Closing them outright
//! could result in their file descriptors being reused. This causes problems
//! for library code (including the standard library) that expects these file
//! descriptors to have special meaning.
//!
//! This function discards the standard input and standard output streams by
//! opening `/dev/null` and assigning it to their file descriptors, closing
//! whatever had been at those file descriptors previously.
//!
//! `stderr`, the standard error stream, is not closed. It is often useful to
//! retain the ability to send diagnostic messages to the standard error stream.
//!
//! \note This function can only maintain its guarantees in a single-threaded
//!     process, or in situations where the caller has control of all threads in
//!     the process.
void CloseStdinAndStdout();

}  // namespace crashpad

#endif  // CRASHPAD_UTIL_POSIX_CLOSE_STDIO_H_
