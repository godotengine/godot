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

#include "util/win/command_line.h"

#include <sys/types.h>

namespace crashpad {

// Ref:
// https://blogs.msdn.microsoft.com/twistylittlepassagesallalike/2011/04/23/everyone-quotes-command-line-arguments-the-wrong-way/
void AppendCommandLineArgument(const std::wstring& argument,
                               std::wstring* command_line) {
  if (!command_line->empty()) {
    command_line->push_back(L' ');
  }

  // Don’t bother quoting if unnecessary.
  if (!argument.empty() &&
      argument.find_first_of(L" \t\n\v\"") == std::wstring::npos) {
    command_line->append(argument);
  } else {
    command_line->push_back(L'"');
    for (std::wstring::const_iterator i = argument.begin();; ++i) {
      size_t backslash_count = 0;
      while (i != argument.end() && *i == L'\\') {
        ++i;
        ++backslash_count;
      }
      if (i == argument.end()) {
        // Escape all backslashes, but let the terminating double quotation mark
        // we add below be interpreted as a metacharacter.
        command_line->append(backslash_count * 2, L'\\');
        break;
      } else if (*i == L'"') {
        // Escape all backslashes and the following double quotation mark.
        command_line->append(backslash_count * 2 + 1, L'\\');
        command_line->push_back(*i);
      } else {
        // Backslashes aren’t special here.
        command_line->append(backslash_count, L'\\');
        command_line->push_back(*i);
      }
    }
    command_line->push_back(L'"');
  }
}

}  // namespace crashpad
