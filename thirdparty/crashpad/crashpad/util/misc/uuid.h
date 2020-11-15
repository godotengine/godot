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

#ifndef CRASHPAD_UTIL_MISC_UUID_H_
#define CRASHPAD_UTIL_MISC_UUID_H_

#include <stdint.h>

#include <string>

#include "base/strings/string16.h"
#include "base/strings/string_piece.h"
#include "build/build_config.h"

#if defined(OS_WIN)
#include <rpc.h>
#endif

namespace crashpad {

//! \brief A universally unique identifier (%UUID).
//!
//! An alternate term for %UUID is “globally unique identifier” (GUID), used
//! primarily by Microsoft.
//!
//! A %UUID is a unique 128-bit number specified by RFC 4122.
//!
//! This is a POD structure.
struct UUID {
  bool operator==(const UUID& that) const;
  bool operator!=(const UUID& that) const { return !operator==(that); }

  //! \brief Initializes the %UUID to zero.
  void InitializeToZero();

  //! \brief Initializes the %UUID from a sequence of bytes.
  //!
  //! \a bytes is taken as a %UUID laid out in big-endian format in memory. On
  //! little-endian machines, appropriate byte-swapping will be performed to
  //! initialize an object’s data members.
  //!
  //! \param[in] bytes A buffer of exactly 16 bytes that will be assigned to the
  //!     %UUID.
  void InitializeFromBytes(const uint8_t* bytes);

  //! \brief Initializes the %UUID from a RFC 4122 §3 formatted string.
  //!
  //! \param[in] string A string of the form
  //!     `"00112233-4455-6677-8899-aabbccddeeff"`.
  //!
  //! \return `true` if the string was formatted correctly and the object has
  //!     been initialized with the data. `false` if the string could not be
  //!     parsed, with the object state untouched.
  bool InitializeFromString(const base::StringPiece& string);
  bool InitializeFromString(const base::StringPiece16& string);

  //! \brief Initializes the %UUID using a standard system facility to generate
  //!     the value.
  //!
  //! \return `true` if the %UUID was initialized correctly, `false` otherwise
  //!     with a message logged.
  bool InitializeWithNew();

#if defined(OS_WIN) || DOXYGEN
  //! \brief Initializes the %UUID from a system `UUID` or `GUID` structure.
  //!
  //! \param[in] system_uuid A system `UUID` or `GUID` structure.
  void InitializeFromSystemUUID(const ::UUID* system_uuid);
#endif  // OS_WIN

  //! \brief Formats the %UUID per RFC 4122 §3.
  //!
  //! \return A string of the form `"00112233-4455-6677-8899-aabbccddeeff"`.
  std::string ToString() const;

#if defined(OS_WIN) || DOXYGEN
  //! \brief The same as ToString, but returned as a string16.
  base::string16 ToString16() const;
#endif  // OS_WIN

  // These fields are laid out according to RFC 4122 §4.1.2.
  uint32_t data_1;
  uint16_t data_2;
  uint16_t data_3;
  uint8_t data_4[2];
  uint8_t data_5[6];
};

}  // namespace crashpad

#endif  // CRASHPAD_UTIL_MISC_UUID_H_
