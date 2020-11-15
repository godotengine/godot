// Copyright 2017 The Crashpad Authors. All rights reserved.
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

#ifndef CRASHPAD_UTIL_LINUX_MEMORY_MAP_H_
#define CRASHPAD_UTIL_LINUX_MEMORY_MAP_H_

#include <sys/types.h>

#include <string>
#include <vector>

#include "util/linux/address_types.h"
#include "util/linux/checked_linux_address_range.h"
#include "util/linux/ptrace_connection.h"
#include "util/misc/initialization_state_dcheck.h"

namespace crashpad {

//! \brief Accesses information about mapped memory in another process.
//!
//! The target process must be stopped to guarantee correct mappings. If the
//! target process is not stopped, mappings may be invalid after the return from
//! Initialize(), and even mappings existing at the time Initialize() was called
//! may not be found.
class MemoryMap {
 public:
  //! \brief Information about a mapped region of memory.
  struct Mapping {
    Mapping();
    bool Equals(const Mapping& other) const;

    std::string name;
    CheckedLinuxAddressRange range;
    off_t offset;
    dev_t device;
    ino_t inode;
    bool readable;
    bool writable;
    bool executable;
    bool shareable;
  };

  MemoryMap();
  ~MemoryMap();

  //! \brief Initializes this object with information about the mapped memory
  //!     regions in the process connected via \a connection.
  //!
  //! This method must be called successfully prior to calling any other method
  //! in this class. This method may only be called once.
  //!
  //! \param[in] connection A connection to the process create a map for.
  //!
  //! \return `true` on success, `false` on failure with a message logged.
  bool Initialize(PtraceConnection* connection);

  //! \return The Mapping containing \a address or `nullptr` if no match is
  //!     found. The caller does not take ownership of this object. It is scoped
  //!     to the lifetime of the MemoryMap object that it was obtained from.
  const Mapping* FindMapping(LinuxVMAddress address) const;

  //! \return The Mapping with the lowest base address whose name is \a name or
  //!     `nullptr` if no match is found. The caller does not take ownership of
  //!     this object. It is scoped to the lifetime of the MemoryMap object that
  //!     it was obtained from.
  const Mapping* FindMappingWithName(const std::string& name) const;

  //! \brief Find possible initial mappings of files mapped over several
  //!     segments.
  //!
  //! Executables and libaries are typically loaded into several mappings with
  //! varying permissions for different segments. Portions of an ELF file may
  //! be mapped multiple times as part of loading the file, for example, when
  //! initializing GNU_RELRO segments.
  //!
  //! This method searches for mappings at or below \a mapping in memory that
  //! are mapped from the same file as \a mapping from offset 0.
  //!
  //! On Android, ELF modules may be loaded from within a zipfile, so this
  //! method may return mappings whose offset is not 0.
  //!
  //! This method is intended to help identify the possible base address for
  //! loaded modules, but it is the caller's responsibility to determine which
  //! returned mapping is correct.
  //!
  //! If \a mapping does not refer to a valid mapping, an empty vector will be
  //! returned and a message will be logged. If \a mapping is found but does not
  //! map a file, \a mapping is returned in \a possible_starts.
  //!
  //! \param[in] mapping A Mapping whose series to find the start of.
  //! \return a vector of the possible mapping starts.
  std::vector<const Mapping*> FindFilePossibleMmapStarts(
      const Mapping& mapping) const;

 private:
  std::vector<Mapping> mappings_;
  InitializationStateDcheck initialized_;
};

}  // namespace crashpad

#endif  // CRASHPAD_UTIL_LINUX_MEMORY_MAP_H_
