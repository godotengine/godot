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

#ifndef CRASHPAD_MINIDUMP_MINIDUMP_MODULE_CRASHPAD_INFO_WRITER_H_
#define CRASHPAD_MINIDUMP_MINIDUMP_MODULE_CRASHPAD_INFO_WRITER_H_

#include <stdint.h>
#include <sys/types.h>

#include <memory>
#include <vector>

#include "base/macros.h"
#include "minidump/minidump_extensions.h"
#include "minidump/minidump_string_writer.h"
#include "minidump/minidump_writable.h"

namespace crashpad {

class MinidumpAnnotationListWriter;
class MinidumpSimpleStringDictionaryWriter;
class ModuleSnapshot;

//! \brief The writer for a MinidumpModuleCrashpadInfo object in a minidump
//!     file.
class MinidumpModuleCrashpadInfoWriter final
    : public internal::MinidumpWritable {
 public:
  MinidumpModuleCrashpadInfoWriter();
  ~MinidumpModuleCrashpadInfoWriter() override;

  //! \brief Initializes MinidumpModuleCrashpadInfo based on \a module_snapshot.
  //!
  //! Only data in \a module_snapshot that is considered useful will be
  //! included. For simple annotations, usefulness is determined by
  //! MinidumpSimpleStringDictionaryWriter::IsUseful().
  //!
  //! \param[in] module_snapshot The module snapshot to use as source data.
  //!
  //! \note Valid in #kStateMutable. No mutator methods may be called before
  //!     this method, and it is not normally necessary to call any mutator
  //!     methods after this method.
  void InitializeFromSnapshot(const ModuleSnapshot* module_snapshot);

  //! \brief Arranges for MinidumpModuleCrashpadInfo::list_annotations to point
  //!     to the internal::MinidumpUTF8StringListWriter object to be written by
  //!     \a list_annotations.
  //!
  //! This object takes ownership of \a simple_annotations and becomes its
  //! parent in the overall tree of internal::MinidumpWritable objects.
  //!
  //! \note Valid in #kStateMutable.
  void SetListAnnotations(
      std::unique_ptr<MinidumpUTF8StringListWriter> list_annotations);

  //! \brief Arranges for MinidumpModuleCrashpadInfo::simple_annotations to
  //!     point to the MinidumpSimpleStringDictionaryWriter object to be written
  //!     by \a simple_annotations.
  //!
  //! This object takes ownership of \a simple_annotations and becomes its
  //! parent in the overall tree of internal::MinidumpWritable objects.
  //!
  //! \note Valid in #kStateMutable.
  void SetSimpleAnnotations(
      std::unique_ptr<MinidumpSimpleStringDictionaryWriter> simple_annotations);

  //! \brief Arranges for MinidumpModuleCrashpadInfo::annotation_objects to
  //!     point to the MinidumpAnnotationListWriter object to be written by
  //!     \a annotation_objects.
  //!
  //! This object takes ownership of \a annotation_objects and becomes its
  //! parent in the overall tree of internal::MinidumpWritable objects.
  //!
  //! \note Valid in #kStateMutable.
  void SetAnnotationObjects(
      std::unique_ptr<MinidumpAnnotationListWriter> annotation_objects);

  //! \brief Determines whether the object is useful.
  //!
  //! A useful object is one that carries data that makes a meaningful
  //! contribution to a minidump file. An object carrying list annotations or
  //! simple annotations would be considered useful.
  //!
  //! \return `true` if the object is useful, `false` otherwise.
  bool IsUseful() const;

 protected:
  // MinidumpWritable:
  bool Freeze() override;
  size_t SizeOfObject() override;
  std::vector<MinidumpWritable*> Children() override;
  bool WriteObject(FileWriterInterface* file_writer) override;

 private:
  MinidumpModuleCrashpadInfo module_;
  std::unique_ptr<MinidumpUTF8StringListWriter> list_annotations_;
  std::unique_ptr<MinidumpSimpleStringDictionaryWriter> simple_annotations_;
  std::unique_ptr<MinidumpAnnotationListWriter> annotation_objects_;

  DISALLOW_COPY_AND_ASSIGN(MinidumpModuleCrashpadInfoWriter);
};

//! \brief The writer for a MinidumpModuleCrashpadInfoList object in a minidump
//!     file, containing a list of MinidumpModuleCrashpadInfo objects.
class MinidumpModuleCrashpadInfoListWriter final
    : public internal::MinidumpWritable {
 public:
  MinidumpModuleCrashpadInfoListWriter();
  ~MinidumpModuleCrashpadInfoListWriter() override;

  //! \brief Adds an initialized MinidumpModuleCrashpadInfo for modules in \a
  //!     module_snapshots to the MinidumpModuleCrashpadInfoList.
  //!
  //! Only modules in \a module_snapshots that would produce a useful
  //! MinidumpModuleCrashpadInfo structure are included. Usefulness is
  //! determined by MinidumpModuleCrashpadInfoWriter::IsUseful().
  //!
  //! \param[in] module_snapshots The module snapshots to use as source data.
  //!
  //! \note Valid in #kStateMutable. AddModule() may not be called before this
  //!     method, and it is not normally necessary to call AddModule() after
  //!     this method.
  void InitializeFromSnapshot(
      const std::vector<const ModuleSnapshot*>& module_snapshots);

  //! \brief Adds a MinidumpModuleCrashpadInfo to the
  //!     MinidumpModuleCrashpadInfoList.
  //!
  //! \param[in] module_crashpad_info Extended Crashpad-specific information
  //!     about the module. This object takes ownership of \a
  //!     module_crashpad_info and becomes its parent in the overall tree of
  //!     internal::MinidumpWritable objects.
  //! \param[in] minidump_module_list_index The index of the MINIDUMP_MODULE in
  //!     the minidump file’s MINIDUMP_MODULE_LIST stream that corresponds to \a
  //!     module_crashpad_info.
  //!
  //! \note Valid in #kStateMutable.
  void AddModule(
      std::unique_ptr<MinidumpModuleCrashpadInfoWriter> module_crashpad_info,
      size_t minidump_module_list_index);

  //! \brief Determines whether the object is useful.
  //!
  //! A useful object is one that carries data that makes a meaningful
  //! contribution to a minidump file. An object carrying children would be
  //! considered useful.
  //!
  //! \return `true` if the object is useful, `false` otherwise.
  bool IsUseful() const;

 protected:
  // MinidumpWritable:
  bool Freeze() override;
  size_t SizeOfObject() override;
  std::vector<MinidumpWritable*> Children() override;
  bool WriteObject(FileWriterInterface* file_writer) override;

 private:
  std::vector<std::unique_ptr<MinidumpModuleCrashpadInfoWriter>>
      module_crashpad_infos_;
  std::vector<MinidumpModuleCrashpadInfoLink> module_crashpad_info_links_;
  MinidumpModuleCrashpadInfoList module_crashpad_info_list_base_;

  DISALLOW_COPY_AND_ASSIGN(MinidumpModuleCrashpadInfoListWriter);
};

}  // namespace crashpad

#endif  // CRASHPAD_MINIDUMP_MINIDUMP_MODULE_CRASHPAD_INFO_WRITER_H_
