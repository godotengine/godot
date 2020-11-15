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

#ifndef CRASHPAD_UTIL_WIN_COMMAND_LINE_H_
#define CRASHPAD_UTIL_WIN_COMMAND_LINE_H_

#include <string>

namespace crashpad {

//! \brief Utility function for building escaped command lines.
//!
//! This builds a command line so that individual arguments can be reliably
//! decoded by `CommandLineToArgvW()`.
//!
//! \a argument is appended to \a command_line. If necessary, it will be placed
//! in quotation marks and escaped properly. If \a command_line is initially
//! non-empty, a space will precede \a argument.
//!
//! \param[in] argument The argument to append to \a command_line.
//! \param[in,out] command_line The command line being constructed.
void AppendCommandLineArgument(const std::wstring& argument,
                               std::wstring* command_line);

}  // namespace crashpad

#endif  // CRASHPAD_UTIL_WIN_COMMAND_LINE_H_
