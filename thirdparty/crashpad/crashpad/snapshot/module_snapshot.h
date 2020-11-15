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

#ifndef CRASHPAD_SNAPSHOT_MODULE_SNAPSHOT_H_
#define CRASHPAD_SNAPSHOT_MODULE_SNAPSHOT_H_

#include <stdint.h>
#include <sys/types.h>

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "snapshot/annotation_snapshot.h"
#include "snapshot/memory_snapshot.h"
#include "util/misc/uuid.h"
#include "util/numeric/checked_range.h"

namespace crashpad {

class MemorySnapshot;

//! \brief Information describing a custom user data stream in a minidump.
class UserMinidumpStream {
 public:
  //! \brief Constructs a UserMinidumpStream, takes ownership of \a memory.
  UserMinidumpStream(uint32_t stream_type, MemorySnapshot* memory)
      : memory_(memory), stream_type_(stream_type) {}

  const MemorySnapshot* memory() const { return memory_.get(); }
  uint32_t stream_type() const { return stream_type_; }

 private:
  //! \brief The memory representing the custom minidump stream.
  std::unique_ptr<MemorySnapshot> memory_;

  //! \brief The stream type that the minidump stream will be tagged with.
  uint32_t stream_type_;

  DISALLOW_COPY_AND_ASSIGN(UserMinidumpStream);
};

//! \brief An abstract interface to a snapshot representing a code module
//!     (binary image) loaded into a snapshot process.
class ModuleSnapshot {
 public:
  virtual ~ModuleSnapshot() {}

  //! \brief A module’s type.
  enum ModuleType {
    //! \brief The module’s type is unknown.
    kModuleTypeUnknown = 0,

    //! \brief The module is a main executable.
    kModuleTypeExecutable,

    //! \brief The module is a shared library.
    //!
    //! \sa kModuleTypeLoadableModule
    kModuleTypeSharedLibrary,

    //! \brief The module is a loadable module.
    //!
    //! On some platforms, loadable modules are distinguished from shared
    //! libraries. On these platforms, a shared library is a module that another
    //! module links against directly, and a loadable module is not. Loadable
    //! modules tend to be binary plug-ins.
    kModuleTypeLoadableModule,

    //! \brief The module is a dynamic loader.
    //!
    //! This is the module responsible for loading other modules. This is
    //! normally `dyld` for macOS and `ld.so` for Linux and other systems using
    //! ELF.
    kModuleTypeDynamicLoader,
  };

  //! \brief Returns the module’s pathname.
  virtual std::string Name() const = 0;

  //! \brief Returns the base address that the module is loaded at in the
  //!     snapshot process.
  virtual uint64_t Address() const = 0;

  //! \brief Returns the size that the module occupies in the snapshot process’
  //!     address space, starting at its base address.
  //!
  //! For macOS snapshots, this method only reports the size of the `__TEXT`
  //! segment, because segments may not be loaded contiguously.
  virtual uint64_t Size() const = 0;

  //! \brief Returns the module’s timestamp, if known.
  //!
  //! The timestamp is typically the modification time of the file that provided
  //! the module in `time_t` format, seconds since the POSIX epoch. If the
  //! module’s timestamp is unknown, this method returns `0`.
  virtual time_t Timestamp() const = 0;

  //! \brief Returns the module’s file version in the \a version_* parameters.
  //!
  //! If no file version can be determined, the \a version_* parameters are set
  //! to `0`.
  //!
  //! For macOS snapshots, this is taken from the module’s `LC_ID_DYLIB` load
  //! command for shared libraries, and is `0` for other module types.
  virtual void FileVersion(uint16_t* version_0,
                           uint16_t* version_1,
                           uint16_t* version_2,
                           uint16_t* version_3) const = 0;

  //! \brief Returns the module’s source version in the \a version_* parameters.
  //!
  //! If no source version can be determined, the \a version_* parameters are
  //! set to `0`.
  //!
  //! For macOS snapshots, this is taken from the module’s `LC_SOURCE_VERSION`
  //! load command.
  virtual void SourceVersion(uint16_t* version_0,
                             uint16_t* version_1,
                             uint16_t* version_2,
                             uint16_t* version_3) const = 0;

  //! \brief Returns the module’s type.
  virtual ModuleType GetModuleType() const = 0;

  //! \brief Returns the module’s UUID in the \a uuid parameter, and the age of
  //!     that UUID in \a age.
  //!
  //! A snapshot module’s UUID is taken directly from the module itself. If the
  //! module does not have a UUID, the \a uuid parameter will be zeroed out.
  //!
  //! \a age is the number of times the UUID has been reused. This occurs on
  //! Windows with incremental linking. On other platforms \a age will always be
  //! `0`.
  //!
  //! \sa DebugFileName()
  virtual void UUIDAndAge(crashpad::UUID* uuid, uint32_t* age) const = 0;

  //! \brief Returns the module’s debug file info name.
  //!
  //! On Windows, this references the PDB file, which contains symbol
  //! information held separately from the module itself. On other platforms,
  //! this is normally the basename of the module, because the debug info file’s
  //! name is not relevant even in split-debug scenarios.
  //!
  //! \sa UUIDAndAge()
  virtual std::string DebugFileName() const = 0;

  //! \brief Returns string annotations recorded in the module.
  //!
  //! This method retrieves annotations recorded in a module. These annotations
  //! are intended for diagnostic use, including crash analysis. A module may
  //! contain multiple annotations, so they are returned in a vector.
  //!
  //! For macOS snapshots, these annotations are found by interpreting the
  //! module’s `__DATA,__crash_info` section as `crashreporter_annotations_t`.
  //! System libraries using the crash reporter client interface may reference
  //! annotations in this structure. Additional annotations messages may be
  //! found in other locations, which may be module-specific. The dynamic linker
  //! (`dyld`) can provide an annotation at its `_error_string` symbol.
  //!
  //! The annotations returned by this method do not duplicate those returned by
  //! AnnotationsSimpleMap() or AnnotationObjects().
  virtual std::vector<std::string> AnnotationsVector() const = 0;

  //! \brief Returns key-value string annotations recorded in the module.
  //!
  //! This method retrieves annotations recorded in a module. These annotations
  //! are intended for diagnostic use, including crash analysis. “Simple
  //! annotations” are structured as a sequence of key-value pairs, where all
  //! keys and values are strings. These are referred to in Chrome as “crash
  //! keys.”
  //!
  //! For macOS snapshots, these annotations are found by interpreting the
  //! `__DATA,crashpad_info` section as `CrashpadInfo`. Clients can use the
  //! Crashpad client interface to store annotations in this structure. Most
  //! annotations under the client’s direct control will be retrievable by this
  //! method. For clients such as Chrome, this includes the process type.
  //!
  //! The annotations returned by this method do not duplicate those returned by
  //! AnnotationsVector() or AnnotationObjects(). Additional annotations related
  //! to the process, system, or snapshot producer may be obtained by calling
  //! ProcessSnapshot::AnnotationsSimpleMap().
  virtual std::map<std::string, std::string> AnnotationsSimpleMap() const = 0;

  //! \brief Returns the typed annotation objects recorded in the module.
  //!
  //! This method retrieves annotations recorded in a module. These annotations
  //! are intended for diagnostic use, including crash analysis. Annotation
  //! objects are strongly-typed name-value pairs. The names are not unique.
  //!
  //! For macOS snapshots, these annotations are found by interpreting the
  //! `__DATA,crashpad_info` section as `CrashpadInfo`. Clients can use the
  //! Crashpad client interface to store annotations in this structure. Most
  //! annotations under the client’s direct control will be retrievable by this
  //! method. For clients such as Chrome, this includes the process type.
  //!
  //! The annotations returned by this method do not duplicate those returned by
  //! AnnotationsVector() or AnnotationsSimpleMap().
  virtual std::vector<AnnotationSnapshot> AnnotationObjects() const = 0;

  //! \brief Returns a set of extra memory ranges specified in the module as
  //!     being desirable to include in the crash dump.
  virtual std::set<CheckedRange<uint64_t>> ExtraMemoryRanges() const = 0;

  //! \brief Returns a list of custom minidump stream specified in the module to
  //!     be included in the crash dump.
  //!
  //! \return The caller does not take ownership of the returned objects, they
  //!     are scoped to the lifetime of the ModuleSnapshot object that they were
  //!     obtained from.
  virtual std::vector<const UserMinidumpStream*> CustomMinidumpStreams()
      const = 0;
};

}  // namespace crashpad

#endif  // CRASHPAD_SNAPSHOT_MODULE_SNAPSHOT_H_
