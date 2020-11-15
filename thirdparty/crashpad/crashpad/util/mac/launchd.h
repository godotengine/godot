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

#ifndef CRASHPAD_UTIL_MAC_LAUNCHD_H_
#define CRASHPAD_UTIL_MAC_LAUNCHD_H_

#include <CoreFoundation/CoreFoundation.h>
#include <launch.h>
#include <sys/types.h>

namespace crashpad {

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

//! \{
//! \brief Wraps the `<launch.h>` function of the same name.
//!
//! The OS X 10.10 SDK deprecates `<launch.h>`, although the functionality it
//! provides is still useful. These wrappers allow the deprecated functions to
//! be called without triggering deprecated-declaration warnings.

inline launch_data_t LaunchDataAlloc(launch_data_type_t type) {
  return launch_data_alloc(type);
}

inline launch_data_type_t LaunchDataGetType(const launch_data_t data) {
  return launch_data_get_type(data);
}

inline void LaunchDataFree(launch_data_t data) {
  return launch_data_free(data);
}

inline bool LaunchDataDictInsert(launch_data_t dict,
                                 const launch_data_t value,
                                 const char* key) {
  return launch_data_dict_insert(dict, value, key);
}

inline launch_data_t LaunchDataDictLookup(const launch_data_t dict,
                                          const char* key) {
  return launch_data_dict_lookup(dict, key);
}

inline size_t LaunchDataDictGetCount(launch_data_t dict) {
  return launch_data_dict_get_count(dict);
}

inline bool LaunchDataArraySetIndex(launch_data_t array,
                                    const launch_data_t value,
                                    size_t index) {
  return launch_data_array_set_index(array, value, index);
}

inline launch_data_t LaunchDataArrayGetIndex(launch_data_t array,
                                             size_t index) {
  return launch_data_array_get_index(array, index);
}

inline size_t LaunchDataArrayGetCount(launch_data_t array) {
  return launch_data_array_get_count(array);
}

inline launch_data_t LaunchDataNewInteger(long long integer) {
  return launch_data_new_integer(integer);
}

inline launch_data_t LaunchDataNewBool(bool boolean) {
  return launch_data_new_bool(boolean);
}

inline launch_data_t LaunchDataNewReal(double real) {
  return launch_data_new_real(real);
}

inline launch_data_t LaunchDataNewString(const char* string) {
  return launch_data_new_string(string);
}

inline launch_data_t LaunchDataNewOpaque(const void* opaque, size_t size) {
  return launch_data_new_opaque(opaque, size);
}

inline long long LaunchDataGetInteger(const launch_data_t data) {
  return launch_data_get_integer(data);
}

inline bool LaunchDataGetBool(const launch_data_t data) {
  return launch_data_get_bool(data);
}

inline double LaunchDataGetReal(const launch_data_t data) {
  return launch_data_get_real(data);
}

inline const char* LaunchDataGetString(const launch_data_t data) {
  return launch_data_get_string(data);
}

inline void* LaunchDataGetOpaque(const launch_data_t data) {
  return launch_data_get_opaque(data);
}

inline size_t LaunchDataGetOpaqueSize(const launch_data_t data) {
  return launch_data_get_opaque_size(data);
}

inline int LaunchDataGetErrno(const launch_data_t data) {
  return launch_data_get_errno(data);
}

inline launch_data_t LaunchMsg(const launch_data_t data) {
  return launch_msg(data);
}

//! \}

#pragma clang diagnostic pop

//! \brief Converts a Core Foundation-type property list to a launchd-type
//!     `launch_data_t`.
//!
//! \param[in] property_cf The Core Foundation-type property list to convert.
//!
//! \return The converted launchd-type `launch_data_t`. The caller takes
//!     ownership of the returned value. On error, returns `nullptr`.
//!
//! \note This function handles all `CFPropertyListRef` types except for
//!     `CFDateRef`, because there’s no `launch_data_type_t` analogue. Not all
//!     types supported in a launchd-type `launch_data_t` have
//!     `CFPropertyListRef` analogues.
launch_data_t CFPropertyToLaunchData(CFPropertyListRef property_cf);

}  // namespace crashpad

#endif  // CRASHPAD_UTIL_MAC_LAUNCHD_H_
