// Copyright 2018 The Crashpad Authors. All rights reserved.
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

#ifndef CRASHPAD_CLIENT_CLIENT_ARGV_HANDLING_H_
#define CRASHPAD_CLIENT_CLIENT_ARGV_HANDLING_H_

#include <map>
#include <string>
#include <vector>

#include "base/files/file_path.h"

namespace crashpad {

//! \brief Builds a vector of arguments suitable for invoking a handler process
//!     based on arguments passed to StartHandler-type().
//!
//! See StartHandlerAtCrash() for documentation on the input arguments.
//!
//! \return A vector of arguments suitable for starting the handler with.
std::vector<std::string> BuildHandlerArgvStrings(
    const base::FilePath& handler,
    const base::FilePath& database,
    const base::FilePath& metrics_dir,
    const std::string& url,
    const std::map<std::string, std::string>& annotations,
    const std::vector<std::string>& arguments);

//! \brief Flattens a string vector into a const char* vector suitable for use
//!     in an exec() call.
//!
//! \param[in] strings A vector of string data. This vector must remain valid
//!     for the lifetime of \a c_strings.
//! \param[out] c_strings A vector of pointers to the string data in \a strings.
void StringVectorToCStringVector(const std::vector<std::string>& strings,
                                 std::vector<const char*>* c_strings);

}  // namespace crashpad

#endif  // CRASHPAD_CLIENT_CLIENT_ARGV_HANDLING_H_
