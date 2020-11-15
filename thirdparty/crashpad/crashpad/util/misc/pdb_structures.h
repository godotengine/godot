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

#ifndef CRASHPAD_UTIL_MISC_PDB_STRUCTURES_H_
#define CRASHPAD_UTIL_MISC_PDB_STRUCTURES_H_

#include <stdint.h>

#include "util/misc/uuid.h"

namespace crashpad {

//! \brief A CodeView record linking to a `.pdb` 2.0 file.
//!
//! This format provides an indirect link to debugging data by referencing an
//! external `.pdb` file by its name, timestamp, and age. This structure may be
//! pointed to by MINIDUMP_MODULE::CvRecord. It has been superseded by
//! CodeViewRecordPDB70.
//!
//! For more information about this structure and format, see <a
//! href="http://www.debuginfo.com/articles/debuginfomatch.html#pdbfiles">Matching
//! Debug Information</a>, PDB Files, and <i>Undocumented Windows 2000
//! Secrets</i>, Windows 2000 Debugging Support/Microsoft Symbol File
//! Internals/CodeView Subsections.
//!
//! \sa IMAGE_DEBUG_MISC
struct CodeViewRecordPDB20 {
  //! \brief The magic number identifying this structure version, stored in
  //!     #signature.
  //!
  //! In a hex dump, this will appear as “NB10” when produced by a little-endian
  //! machine.
  static const uint32_t kSignature = '01BN';

  //! \brief The magic number identifying this structure version, the value of
  //!     #kSignature.
  uint32_t signature;

  //! \brief The offset to CodeView data.
  //!
  //! In this structure, this field always has the value `0` because no CodeView
  //! data is present, there is only a link to CodeView data stored in an
  //! external file.
  uint32_t offset;

  //! \brief The time that the `.pdb` file was created, in `time_t` format, the
  //!     number of seconds since the POSIX epoch.
  uint32_t timestamp;

  //! \brief The revision of the `.pdb` file.
  //!
  //! A `.pdb` file’s age indicates incremental changes to it. When a `.pdb`
  //! file is created, it has age `1`, and subsequent updates increase this
  //! value.
  uint32_t age;

  //! \brief The path or file name of the `.pdb` file associated with the
  //!     module.
  //!
  //! This is a NUL-terminated string. On Windows, it will be encoded in the
  //! code page of the system that linked the module. On other operating
  //! systems, UTF-8 may be used.
  uint8_t pdb_name[1];
};

//! \brief A CodeView record linking to a `.pdb` 7.0 file.
//!
//! This format provides an indirect link to debugging data by referencing an
//! external `.pdb` file by its name, %UUID, and age. This structure may be
//! pointed to by MINIDUMP_MODULE::CvRecord.
//!
//! For more information about this structure and format, see <a
//! href="http://www.debuginfo.com/articles/debuginfomatch.html#pdbfiles">Matching
//! Debug Information</a>, PDB Files.
//!
//! \sa CodeViewRecordPDB20
//! \sa IMAGE_DEBUG_MISC
struct CodeViewRecordPDB70 {
  // UUID has a constructor, which makes it non-POD, which makes this structure
  // non-POD. In order for the default constructor to zero-initialize other
  // members, an explicit constructor must be provided.
  CodeViewRecordPDB70()
      : signature(),
        uuid(),
        age(),
        pdb_name() {
  }

  //! \brief The magic number identifying this structure version, stored in
  //!     #signature.
  //!
  //! In a hex dump, this will appear as “RSDS” when produced by a little-endian
  //! machine.
  static const uint32_t kSignature = 'SDSR';

  //! \brief The magic number identifying this structure version, the value of
  //!     #kSignature.
  uint32_t signature;

  //! \brief The `.pdb` file’s unique identifier.
  UUID uuid;

  //! \brief The revision of the `.pdb` file.
  //!
  //! A `.pdb` file’s age indicates incremental changes to it. When a `.pdb`
  //! file is created, it has age `1`, and subsequent updates increase this
  //! value.
  uint32_t age;

  //! \brief The path or file name of the `.pdb` file associated with the
  //!     module.
  //!
  //! This is a NUL-terminated string. On Windows, it will be encoded in the
  //! code page of the system that linked the module. On other operating
  //! systems, UTF-8 may be used.
  uint8_t pdb_name[1];
};

}  // namespace crashpad

#endif  // CRASHPAD_UTIL_MISC_PDB_STRUCTURES_H_
