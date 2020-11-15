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

#include "util/posix/symbolic_constants_posix.h"

#include <signal.h>
#include <string.h>
#include <sys/types.h>

#include "base/macros.h"
#include "base/strings/stringprintf.h"
#include "build/build_config.h"
#include "util/misc/implicit_cast.h"
#include "util/stdlib/string_number_conversion.h"

namespace {

constexpr const char* kSignalNames[] = {
    nullptr,

#if defined(OS_MACOSX)
    // sed -Ene 's/^#define[[:space:]]SIG([[:alnum:]]+)[[:space:]]+[[:digit:]]{1,2}([[:space:]]|$).*/    "\1",/p'
    //     /usr/include/sys/signal.h
    // and fix up by removing the entry for SIGPOLL.
    "HUP",
    "INT",
    "QUIT",
    "ILL",
    "TRAP",
    "ABRT",
    "EMT",
    "FPE",
    "KILL",
    "BUS",
    "SEGV",
    "SYS",
    "PIPE",
    "ALRM",
    "TERM",
    "URG",
    "STOP",
    "TSTP",
    "CONT",
    "CHLD",
    "TTIN",
    "TTOU",
    "IO",
    "XCPU",
    "XFSZ",
    "VTALRM",
    "PROF",
    "WINCH",
    "INFO",
    "USR1",
    "USR2",
#elif defined(OS_LINUX) || defined(OS_ANDROID)
#if defined(ARCH_CPU_MIPS_FAMILY)
    "HUP",
    "INT",
    "QUIT",
    "ILL",
    "TRAP",
    "ABRT",
    "EMT",
    "FPE",
    "KILL",
    "BUS",
    "SEGV",
    "SYS",
    "PIPE",
    "ALRM",
    "TERM",
    "USR1",
    "USR2",
    "CHLD",
    "PWR",
    "WINCH",
    "URG",
    "IO",
    "STOP",
    "TSTP",
    "CONT",
    "TTIN",
    "TTOU",
    "VTALRM",
    "PROF",
    "XCPU",
    "XFSZ",
#else
    // sed -Ene 's/^#define[[:space:]]SIG([[:alnum:]]+)[[:space:]]+[[:digit:]]{1,2}([[:space:]]|$).*/    "\1",/p'
    //     /usr/include/asm-generic/signal.h
    // and fix up by removing SIGIOT, SIGLOST, SIGUNUSED, and SIGRTMIN.
    "HUP",
    "INT",
    "QUIT",
    "ILL",
    "TRAP",
    "ABRT",
    "BUS",
    "FPE",
    "KILL",
    "USR1",
    "SEGV",
    "USR2",
    "PIPE",
    "ALRM",
    "TERM",
    "STKFLT",
    "CHLD",
    "CONT",
    "STOP",
    "TSTP",
    "TTIN",
    "TTOU",
    "URG",
    "XCPU",
    "XFSZ",
    "VTALRM",
    "PROF",
    "WINCH",
    "IO",
    "PWR",
    "SYS",
#endif  // defined(ARCH_CPU_MIPS_FAMILY)
#endif
};
#if defined(OS_LINUX) || defined(OS_ANDROID)
// NSIG is 64 to account for real-time signals.
static_assert(arraysize(kSignalNames) == 32, "kSignalNames length");
#else
static_assert(arraysize(kSignalNames) == NSIG, "kSignalNames length");
#endif

constexpr char kSigPrefix[] = "SIG";

}  // namespace

namespace crashpad {

std::string SignalToString(int signal,
                           SymbolicConstantToStringOptions options) {
  const char* signal_name =
      implicit_cast<size_t>(signal) < arraysize(kSignalNames)
          ? kSignalNames[signal]
          : nullptr;
  if (!signal_name) {
    if (options & kUnknownIsNumeric) {
      return base::StringPrintf("%d", signal);
    }
    return std::string();
  }

  if (options & kUseShortName) {
    return std::string(signal_name);
  }
  return base::StringPrintf("%s%s", kSigPrefix, signal_name);
}

bool StringToSignal(const base::StringPiece& string,
                    StringToSymbolicConstantOptions options,
                    int* signal) {
  if ((options & kAllowFullName) || (options & kAllowShortName)) {
    bool can_match_full =
        (options & kAllowFullName) &&
        string.substr(0, strlen(kSigPrefix)).compare(kSigPrefix) == 0;
    base::StringPiece short_string =
        can_match_full ? string.substr(strlen(kSigPrefix)) : string;
    for (int index = 0;
         index < implicit_cast<int>(arraysize(kSignalNames));
         ++index) {
      const char* signal_name = kSignalNames[index];
      if (!signal_name) {
        continue;
      }
      if (can_match_full && short_string.compare(signal_name) == 0) {
        *signal = index;
        return true;
      }
      if ((options & kAllowShortName) && string.compare(signal_name) == 0) {
        *signal = index;
        return true;
      }
    }
  }

  if (options & kAllowNumber) {
    return StringToNumber(std::string(string.data(), string.length()), signal);
  }

  return false;
}

}  // namespace crashpad
