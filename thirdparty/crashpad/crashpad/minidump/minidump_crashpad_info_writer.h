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

#ifndef CRASHPAD_MINIDUMP_MINIDUMP_CRASHPAD_INFO_WRITER_H_
#define CRASHPAD_MINIDUMP_MINIDUMP_CRASHPAD_INFO_WRITER_H_

#include <sys/types.h>

#include <memory>
#include <vector>

#include "base/macros.h"
#include "minidump/minidump_extensions.h"
#include "minidump/minidump_stream_writer.h"
#include "util/misc/uuid.h"

namespace crashpad {

class MinidumpModuleCrashpadInfoListWriter;
class MinidumpSimpleStringDictionaryWriter;
class ProcessSnapshot;

//! \brief The writer for a MinidumpCrashpadInfo stream in a minidump file.
class MinidumpCrashpadInfoWriter final : public internal::MinidumpStreamWriter {
 public:
  MinidumpCrashpadInfoWriter();
  ~MinidumpCrashpadInfoWriter() override;

  //! \brief Initializes MinidumpCrashpadInfo based on \a process_snapshot.
  //!
  //! This method may add additional structures to the minidump file as children
  //! of the MinidumpCrashpadInfo stream. To do so, it may obtain other
  //! snapshot information from \a process_snapshot, such as a list of
  //! ModuleSnapshot objects used to initialize
  //! MinidumpCrashpadInfo::module_list. Only data that is considered useful
  //! will be included. For module information, usefulness is determined by
  //! MinidumpModuleCrashpadInfoListWriter::IsUseful().
  //!
  //! \param[in] process_snapshot The process snapshot to use as source data.
  //!
  //! \note Valid in #kStateMutable. No mutator methods may be called before
  //!     this method, and it is not normally necessary to call any mutator
  //!     methods after this method.
  void InitializeFromSnapshot(const ProcessSnapshot* process_snapshot);

  //! \brief Sets MinidumpCrashpadInfo::report_id.
  void SetReportID(const UUID& report_id);

  //! \brief Sets MinidumpCrashpadInfo::client_id.
  void SetClientID(const UUID& client_id);

  //! \brief Arranges for MinidumpCrashpadInfo::simple_annotations to point to
  //!     the MinidumpSimpleStringDictionaryWriter object to be written by \a
  //!     simple_annotations.
  //!
  //! This object takes ownership of \a simple_annotations and becomes its
  //! parent in the overall tree of internal::MinidumpWritable objects.
  //!
  //! \note Valid in #kStateMutable.
  void SetSimpleAnnotations(
      std::unique_ptr<MinidumpSimpleStringDictionaryWriter> simple_annotations);

  //! \brief Arranges for MinidumpCrashpadInfo::module_list to point to the
  //!     MinidumpModuleCrashpadInfoList object to be written by \a
  //!     module_list.
  //!
  //! This object takes ownership of \a module_list and becomes its parent in
  //! the overall tree of internal::MinidumpWritable objects.
  //!
  //! \note Valid in #kStateMutable.
  void SetModuleList(
      std::unique_ptr<MinidumpModuleCrashpadInfoListWriter> module_list);

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

  // MinidumpStreamWriter:
  MinidumpStreamType StreamType() const override;

 private:
  MinidumpCrashpadInfo crashpad_info_;
  std::unique_ptr<MinidumpSimpleStringDictionaryWriter> simple_annotations_;
  std::unique_ptr<MinidumpModuleCrashpadInfoListWriter> module_list_;

  DISALLOW_COPY_AND_ASSIGN(MinidumpCrashpadInfoWriter);
};

}  // namespace crashpad

#endif  // CRASHPAD_MINIDUMP_MINIDUMP_CRASHPAD_INFO_WRITER_H_
