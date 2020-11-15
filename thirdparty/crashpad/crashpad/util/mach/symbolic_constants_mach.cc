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

#include "util/mach/symbolic_constants_mach.h"

#include <string.h>
#include <sys/types.h>

#include "base/macros.h"
#include "base/strings/stringprintf.h"
#include "util/mach/exception_behaviors.h"
#include "util/mach/mach_extensions.h"
#include "util/misc/implicit_cast.h"
#include "util/stdlib/string_number_conversion.h"

namespace {

constexpr const char* kExceptionNames[] = {
    nullptr,

    // sed -Ene 's/^#define[[:space:]]EXC_([[:graph:]]+)[[:space:]]+[[:digit:]]{1,2}([[:space:]]|$).*/    "\1",/p'
    //     /usr/include/mach/exception_types.h
    "BAD_ACCESS",
    "BAD_INSTRUCTION",
    "ARITHMETIC",
    "EMULATION",
    "SOFTWARE",
    "BREAKPOINT",
    "SYSCALL",
    "MACH_SYSCALL",
    "RPC_ALERT",
    "CRASH",
    "RESOURCE",
    "GUARD",
    "CORPSE_NOTIFY",
};
static_assert(arraysize(kExceptionNames) == EXC_TYPES_COUNT,
              "kExceptionNames length");

constexpr char kExcPrefix[] = "EXC_";
constexpr char kExcMaskPrefix[] = "EXC_MASK_";

constexpr const char* kBehaviorNames[] = {
    nullptr,

    // sed -Ene 's/^# define[[:space:]]EXCEPTION_([[:graph:]]+)[[:space:]]+[[:digit:]]{1,2}([[:space:]]|$).*/    "\1",/p'
    //     /usr/include/mach/exception_types.h
    "DEFAULT",
    "STATE",
    "STATE_IDENTITY",
};

constexpr char kBehaviorPrefix[] = "EXCEPTION_";
constexpr char kMachExceptionCodesFull[] = "MACH_EXCEPTION_CODES";
constexpr char kMachExceptionCodesShort[] = "MACH";

constexpr const char* kFlavorNames[] = {
    "THREAD_STATE_FLAVOR_LIST",

#if defined(__i386__) || defined(__x86_64__)
    // sed -Ene 's/^#define ((x86|THREAD)_[[:graph:]]+)[[:space:]]+[[:digit:]]{1,2}.*$/    "\1",/p'
    //     /usr/include/mach/i386/thread_status.h
    // and then fix up by adding x86_SAVED_STATE32 and x86_SAVED_STATE64.
    "x86_THREAD_STATE32",
    "x86_FLOAT_STATE32",
    "x86_EXCEPTION_STATE32",
    "x86_THREAD_STATE64",
    "x86_FLOAT_STATE64",
    "x86_EXCEPTION_STATE64",
    "x86_THREAD_STATE",
    "x86_FLOAT_STATE",
    "x86_EXCEPTION_STATE",
    "x86_DEBUG_STATE32",
    "x86_DEBUG_STATE64",
    "x86_DEBUG_STATE",
    "THREAD_STATE_NONE",
    "x86_SAVED_STATE32",
    "x86_SAVED_STATE64",
    "x86_AVX_STATE32",
    "x86_AVX_STATE64",
    "x86_AVX_STATE",
#elif defined(__ppc__) || defined(__ppc64__)
    // sed -Ene 's/^#define ((PPC|THREAD)_[[:graph:]]+)[[:space:]]+[[:digit:]]{1,2}.*$/    "\1",/p'
    //     usr/include/mach/ppc/thread_status.h
    // (Mac OS X 10.6 SDK)
    "PPC_THREAD_STATE",
    "PPC_FLOAT_STATE",
    "PPC_EXCEPTION_STATE",
    "PPC_VECTOR_STATE",
    "PPC_THREAD_STATE64",
    "PPC_EXCEPTION_STATE64",
    "THREAD_STATE_NONE",
#elif defined(__arm__) || defined(__aarch64__)
    // sed -Ene 's/^#define ((ARM|THREAD)_[[:graph:]]+)[[:space:]]+[[:digit:]]{1,2}.*$/    "\1",/p'
    //     usr/include/mach/arm/thread_status.h
    // (iOS 7 SDK)
    // and then fix up by making the list sparse as appropriate.
    "ARM_THREAD_STATE",
    "ARM_VFP_STATE",
    "ARM_EXCEPTION_STATE",
    "ARM_DEBUG_STATE",
    "THREAD_STATE_NONE",
    "ARM_THREAD_STATE64",
    "ARM_EXCEPTION_STATE64",
    nullptr,
    "ARM_THREAD_STATE32",
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    "ARM_DEBUG_STATE32",
    "ARM_DEBUG_STATE64",
    "ARM_NEON_STATE",
    "ARM_NEON_STATE64",
#endif
};

// Certain generic flavors have high constants not contiguous with the flavors
// above. List them separately alongside their constants.
constexpr struct {
  thread_state_flavor_t flavor;
  const char* name;
} kGenericFlavorNames[] = {
    {THREAD_STATE_FLAVOR_LIST_NEW, "THREAD_STATE_FLAVOR_LIST_NEW"},
    {THREAD_STATE_FLAVOR_LIST_10_9, "THREAD_STATE_FLAVOR_LIST_10_9"},
};

// Returns the short name for a flavor name, given its full flavor name.
std::string ThreadStateFlavorFullToShort(const base::StringPiece& flavor) {
  // For generic flavors like THREAD_STATE_NONE and THREAD_STATE_FLAVOR_LIST_*.
  static constexpr char kThreadState[] = "THREAD_STATE_";
  size_t prefix_len = strlen(kThreadState);
  const char* flavor_data = flavor.data();
  size_t flavor_len = flavor.size();
  if (flavor_len >= prefix_len &&
      strncmp(flavor_data, kThreadState, prefix_len) == 0) {
    return std::string(flavor_data + prefix_len, flavor_len - prefix_len);
  }

  // For architecture-specific flavors.
#if defined(__i386__) || defined(__x86_64__)
  static constexpr char kArchPrefix[] = "x86_";
#elif defined(__ppc__) || defined(__ppc64__)
  static constexpr char kArchPrefix[] = "PPC_";
#elif defined(__arm__) || defined(__aarch64__)
  static constexpr char kArchPrefix[] = "ARM_"
#endif
  prefix_len = strlen(kArchPrefix);
  if (flavor_len >= prefix_len &&
      strncmp(flavor_data, kArchPrefix, prefix_len) == 0) {
    // Shorten the suffix by removing _STATE. If the suffix contains a
    // significant designation like 32 or 64, keep it, so that a full name like
    // x86_THREAD_STATE64 becomes a short name like THREAD64.
    static constexpr struct {
      const char* orig;
      const char* repl;
    } kStateSuffixes[] = {
        {"_STATE", ""},
        {"_STATE32", "32"},
        {"_STATE64", "64"},
    };
    for (size_t suffix_index = 0;
         suffix_index < arraysize(kStateSuffixes);
         ++suffix_index) {
      const char* suffix = kStateSuffixes[suffix_index].orig;
      size_t suffix_len = strlen(suffix);
      if (flavor_len >= suffix_len &&
          strncmp(flavor_data + flavor_len - suffix_len, suffix, suffix_len) ==
              0) {
        std::string s(flavor_data + prefix_len,
                      flavor_len - prefix_len - suffix_len);
        return s.append(kStateSuffixes[suffix_index].repl);
      }
    }
  }

  return std::string(flavor_data, flavor_len);
}

}  // namespace

namespace crashpad {

std::string ExceptionToString(exception_type_t exception,
                              SymbolicConstantToStringOptions options) {
  const char* exception_name =
      implicit_cast<size_t>(exception) < arraysize(kExceptionNames)
          ? kExceptionNames[exception]
          : nullptr;
  if (!exception_name) {
    if (options & kUnknownIsNumeric) {
      return base::StringPrintf("%d", exception);
    }
    return std::string();
  }

  if (options & kUseShortName) {
    return std::string(exception_name);
  }
  return base::StringPrintf("%s%s", kExcPrefix, exception_name);
}

bool StringToException(const base::StringPiece& string,
                       StringToSymbolicConstantOptions options,
                       exception_type_t* exception) {
  if ((options & kAllowFullName) || (options & kAllowShortName)) {
    bool can_match_full =
        (options & kAllowFullName) &&
        string.substr(0, strlen(kExcPrefix)).compare(kExcPrefix) == 0;
    base::StringPiece short_string =
        can_match_full ? string.substr(strlen(kExcPrefix)) : string;
    for (exception_type_t index = 0;
         index < implicit_cast<exception_type_t>(arraysize(kExceptionNames));
         ++index) {
      const char* exception_name = kExceptionNames[index];
      if (!exception_name) {
        continue;
      }
      if (can_match_full && short_string.compare(exception_name) == 0) {
        *exception = index;
        return true;
      }
      if ((options & kAllowShortName) && string.compare(exception_name) == 0) {
        *exception = index;
        return true;
      }
    }
  }

  if (options & kAllowNumber) {
    return StringToNumber(std::string(string.data(), string.length()),
                          reinterpret_cast<unsigned int*>(exception));
  }

  return false;
}

std::string ExceptionMaskToString(exception_mask_t exception_mask,
                                  SymbolicConstantToStringOptions options) {
  exception_mask_t local_exception_mask = exception_mask;
  std::string mask_string;
  bool has_forbidden_or = false;
  for (size_t exception = 0;
       exception < arraysize(kExceptionNames);
       ++exception) {
    const char* exception_name = kExceptionNames[exception];
    exception_mask_t exception_mask_value = 1 << exception;
    if (exception_name && (local_exception_mask & exception_mask_value)) {
      if (!mask_string.empty()) {
        if (!(options & kUseOr)) {
          has_forbidden_or = true;
          break;
        }
        mask_string.append("|");
      }
      if (!(options & kUseShortName)) {
        mask_string.append(kExcMaskPrefix);
      }
      mask_string.append(exception_name);
      local_exception_mask &= ~exception_mask_value;
    }
  }

  if (has_forbidden_or) {
    local_exception_mask = exception_mask;
    mask_string.clear();
  }

  // Deal with any remainder.
  if (local_exception_mask) {
    if (!(options & kUnknownIsNumeric)) {
      return std::string();
    }
    if (!mask_string.empty()) {
      mask_string.append("|");
    }
    mask_string.append(base::StringPrintf("%#x", local_exception_mask));
  }

  return mask_string;
}

bool StringToExceptionMask(const base::StringPiece& string,
                           StringToSymbolicConstantOptions options,
                           exception_mask_t* exception_mask) {
  if (options & kAllowOr) {
    options &= ~kAllowOr;
    exception_mask_t build_mask = 0;
    size_t pos = -1;
    do {
      ++pos;
      const char* substring_begin = string.begin() + pos;
      pos = string.find('|', pos);
      const char* substring_end = (pos == base::StringPiece::npos)
                                      ? string.end()
                                      : (string.begin() + pos);
      base::StringPiece substring = string.substr(
          substring_begin - string.begin(), substring_end - substring_begin);

      exception_mask_t temp_mask;
      if (!StringToExceptionMask(substring, options, &temp_mask)) {
        return false;
      }
      build_mask |= temp_mask;
    } while (pos != base::StringPiece::npos);

    *exception_mask = build_mask;
    return true;
  }

  if ((options & kAllowFullName) || (options & kAllowShortName)) {
    bool can_match_full =
        (options & kAllowFullName) &&
        string.substr(0, strlen(kExcMaskPrefix)).compare(kExcMaskPrefix) == 0;
    base::StringPiece short_string =
        can_match_full ? string.substr(strlen(kExcMaskPrefix)) : string;
    for (exception_type_t index = 0;
         index < implicit_cast<exception_type_t>(arraysize(kExceptionNames));
         ++index) {
      const char* exception_name = kExceptionNames[index];
      if (!exception_name) {
        continue;
      }
      if (can_match_full && short_string.compare(exception_name) == 0) {
        *exception_mask = 1 << index;
        return true;
      }
      if ((options & kAllowShortName) && string.compare(exception_name) == 0) {
        *exception_mask = 1 << index;
        return true;
      }
    }

    // EXC_MASK_ALL is a special case: it is not in kExceptionNames as it exists
    // only as a mask value.
    static constexpr char kExcMaskAll[] = "ALL";
    if ((can_match_full && short_string.compare(kExcMaskAll) == 0) ||
        ((options & kAllowShortName) && string.compare(kExcMaskAll) == 0)) {
      *exception_mask = ExcMaskAll();
      return true;
    }
  }

  if (options & kAllowNumber) {
    return StringToNumber(std::string(string.data(), string.length()),
                          reinterpret_cast<unsigned int*>(exception_mask));
  }

  return false;
}

std::string ExceptionBehaviorToString(exception_behavior_t behavior,
                                      SymbolicConstantToStringOptions options) {
  const exception_behavior_t basic_behavior = ExceptionBehaviorBasic(behavior);

  const char* behavior_name =
      implicit_cast<size_t>(basic_behavior) < arraysize(kBehaviorNames)
          ? kBehaviorNames[basic_behavior]
          : nullptr;
  if (!behavior_name) {
    if (options & kUnknownIsNumeric) {
      return base::StringPrintf("%#x", behavior);
    }
    return std::string();
  }

  std::string behavior_string;
  if (options & kUseShortName) {
    behavior_string.assign(behavior_name);
  } else {
    behavior_string.assign(base::StringPrintf(
        "%s%s", kBehaviorPrefix, behavior_name));
  }

  if (ExceptionBehaviorHasMachExceptionCodes(behavior)) {
    behavior_string.append("|");
    behavior_string.append((options & kUseShortName) ? kMachExceptionCodesShort
                                                     : kMachExceptionCodesFull);
  }

  return behavior_string;
}

bool StringToExceptionBehavior(const base::StringPiece& string,
                               StringToSymbolicConstantOptions options,
                               exception_behavior_t* behavior) {
  base::StringPiece sp = string;
  exception_behavior_t build_behavior = 0;
  size_t pos = sp.find('|', 0);
  if (pos != base::StringPiece::npos) {
    base::StringPiece left = sp.substr(0, pos);
    base::StringPiece right = sp.substr(pos + 1, sp.length() - pos - 1);
    if (options & kAllowFullName) {
      if (left.compare(kMachExceptionCodesFull) == 0) {
        build_behavior |= MACH_EXCEPTION_CODES;
        sp = right;
      } else if (right.compare(kMachExceptionCodesFull) == 0) {
        build_behavior |= MACH_EXCEPTION_CODES;
        sp = left;
      }
    }
    if (!(build_behavior & MACH_EXCEPTION_CODES) &&
        (options & kAllowShortName)) {
      if (left.compare(kMachExceptionCodesShort) == 0) {
        build_behavior |= MACH_EXCEPTION_CODES;
        sp = right;
      } else if (right.compare(kMachExceptionCodesShort) == 0) {
        build_behavior |= MACH_EXCEPTION_CODES;
        sp = left;
      }
    }
    if (!(build_behavior & MACH_EXCEPTION_CODES)) {
      return false;
    }
  }

  if ((options & kAllowFullName) || (options & kAllowShortName)) {
    bool can_match_full =
        (options & kAllowFullName) &&
        sp.substr(0, strlen(kBehaviorPrefix)).compare(kBehaviorPrefix) == 0;
    base::StringPiece short_string =
        can_match_full ? sp.substr(strlen(kBehaviorPrefix)) : sp;
    for (exception_behavior_t index = 0;
         index < implicit_cast<exception_behavior_t>(arraysize(kBehaviorNames));
         ++index) {
      const char* behavior_name = kBehaviorNames[index];
      if (!behavior_name) {
        continue;
      }
      if (can_match_full && short_string.compare(behavior_name) == 0) {
        build_behavior |= index;
        *behavior = build_behavior;
        return true;
      }
      if ((options & kAllowShortName) && sp.compare(behavior_name) == 0) {
        build_behavior |= index;
        *behavior = build_behavior;
        return true;
      }
    }
  }

  if (options & kAllowNumber) {
    exception_behavior_t temp_behavior;
    if (!StringToNumber(std::string(sp.data(), sp.length()),
                        reinterpret_cast<unsigned int*>(&temp_behavior))) {
      return false;
    }
    build_behavior |= temp_behavior;
    *behavior = build_behavior;
    return true;
  }

  return false;
}

std::string ThreadStateFlavorToString(thread_state_flavor_t flavor,
                                      SymbolicConstantToStringOptions options) {
  const char* flavor_name =
      implicit_cast<size_t>(flavor) < arraysize(kFlavorNames)
          ? kFlavorNames[flavor]
          : nullptr;

  if (!flavor_name) {
    for (size_t generic_flavor_index = 0;
         generic_flavor_index < arraysize(kGenericFlavorNames);
         ++generic_flavor_index) {
      if (flavor == kGenericFlavorNames[generic_flavor_index].flavor) {
        flavor_name = kGenericFlavorNames[generic_flavor_index].name;
        break;
      }
    }
  }

  if (!flavor_name) {
    if (options & kUnknownIsNumeric) {
      return base::StringPrintf("%d", flavor);
    }
    return std::string();
  }

  if (options & kUseShortName) {
    return ThreadStateFlavorFullToShort(flavor_name);
  }
  return std::string(flavor_name);
}

bool StringToThreadStateFlavor(const base::StringPiece& string,
                               StringToSymbolicConstantOptions options,
                               thread_state_flavor_t* flavor) {
  if ((options & kAllowFullName) || (options & kAllowShortName)) {
    for (thread_state_flavor_t index = 0;
         index < implicit_cast<thread_state_flavor_t>(arraysize(kFlavorNames));
         ++index) {
      const char* flavor_name = kFlavorNames[index];
      if (!flavor_name) {
        continue;
      }
      if ((options & kAllowFullName) && string.compare(flavor_name) == 0) {
        *flavor = index;
        return true;
      }
      if (options & kAllowShortName) {
        std::string short_name = ThreadStateFlavorFullToShort(flavor_name);
        if (string.compare(short_name) == 0) {
          *flavor = index;
          return true;
        }
      }
    }

    for (size_t generic_flavor_index = 0;
         generic_flavor_index < arraysize(kGenericFlavorNames);
         ++generic_flavor_index) {
      const char* flavor_name = kGenericFlavorNames[generic_flavor_index].name;
      thread_state_flavor_t flavor_number =
          kGenericFlavorNames[generic_flavor_index].flavor;
      if ((options & kAllowFullName) && string.compare(flavor_name) == 0) {
        *flavor = flavor_number;
        return true;
      }
      if (options & kAllowShortName) {
        std::string short_name = ThreadStateFlavorFullToShort(flavor_name);
        if (string.compare(short_name) == 0) {
          *flavor = flavor_number;
          return true;
        }
      }
    }
  }

  if (options & kAllowNumber) {
    return StringToNumber(std::string(string.data(), string.length()),
                          reinterpret_cast<unsigned int*>(flavor));
  }

  return false;
}

}  // namespace crashpad
