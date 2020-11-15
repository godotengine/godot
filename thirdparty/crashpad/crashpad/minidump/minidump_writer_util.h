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

#ifndef CRASHPAD_MINIDUMP_MINIDUMP_WRITER_UTIL_H_
#define CRASHPAD_MINIDUMP_MINIDUMP_WRITER_UTIL_H_

#include <stdint.h>
#include <sys/types.h>
#include <time.h>

#include <string>

#include "base/macros.h"
#include "base/strings/string16.h"

namespace crashpad {
namespace internal {

//! \brief A collection of utility functions used by the MinidumpWritable family
//!     of classes.
class MinidumpWriterUtil final {
 public:
  //! \brief Assigns a `time_t` value, logging a warning if the result overflows
  //!     the destination buffer and will be truncated.
  //!
  //! \param[out] destination A pointer to the variable to be assigned to.
  //! \param[in] source The value to assign.
  //!
  //! The minidump format uses `uint32_t` for many timestamp values, but
  //! `time_t` may be wider than this. These year 2038 bugs are a limitation of
  //! the minidump format. An out-of-range error will be noted with a warning,
  //! but is not considered fatal. \a source will be truncated and assigned to
  //! \a destination in this case.
  //!
  //! For `time_t` values with nonfatal overflow semantics, this function is
  //! used in preference to AssignIfInRange(), which fails without performing an
  //! assignment when an out-of-range condition is detected.
  static void AssignTimeT(uint32_t* destination, time_t source);

  //! \brief Converts a UTF-8 string to UTF-16 and returns it. If the string
  //!     cannot be converted losslessly, indicating that the input is not
  //!     well-formed UTF-8, a warning is logged.
  //!
  //! \param[in] utf8 The UTF-8-encoded string to convert.
  //!
  //! \return The \a utf8 string, converted to UTF-16 encoding. If the
  //!     conversion is lossy, U+FFFD “replacement characters” will be
  //!     introduced.
  static base::string16 ConvertUTF8ToUTF16(const std::string& utf8);

  //! \brief Converts a UTF-8 string to UTF-16 and places it into a buffer of
  //!     fixed size, taking care to `NUL`-terminate the buffer and not to
  //!     overflow it. If the string will be truncated or if it cannot be
  //!     converted losslessly, a warning is logged.
  //!
  //! Any unused portion of the \a destination buffer that is not written to by
  //! the converted string will be overwritten with `NUL` UTF-16 code units,
  //! thus, this function always writes \a destination_size `char16` units.
  //!
  //! If the conversion is lossy, U+FFFD “replacement characters” will be
  //! introduced.
  //!
  //! \param[out] destination A pointer to the destination buffer, where the
  //!     UTF-16-encoded string will be written.
  //! \param[in] destination_size The size of \a destination in `char16` units,
  //!     including space used by a `NUL` terminator.
  //! \param[in] source The UTF-8-encoded input string.
  static void AssignUTF8ToUTF16(base::char16* destination,
                                size_t destination_size,
                                const std::string& source);

 private:
  DISALLOW_IMPLICIT_CONSTRUCTORS(MinidumpWriterUtil);
};

}  // namespace internal
}  // namespace crashpad

#endif  // CRASHPAD_MINIDUMP_MINIDUMP_WRITER_UTIL_H_
