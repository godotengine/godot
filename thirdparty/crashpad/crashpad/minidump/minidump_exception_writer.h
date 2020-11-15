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

#ifndef CRASHPAD_MINIDUMP_MINIDUMP_EXCEPTION_WRITER_H_
#define CRASHPAD_MINIDUMP_MINIDUMP_EXCEPTION_WRITER_H_

#include <windows.h>
#include <dbghelp.h>
#include <stdint.h>
#include <sys/types.h>

#include <memory>
#include <vector>

#include "base/macros.h"
#include "minidump/minidump_stream_writer.h"
#include "minidump/minidump_thread_id_map.h"

namespace crashpad {

class ExceptionSnapshot;
class MinidumpContextWriter;
class MinidumpMemoryListWriter;

//! \brief The writer for a MINIDUMP_EXCEPTION_STREAM stream in a minidump file.
class MinidumpExceptionWriter final : public internal::MinidumpStreamWriter {
 public:
  MinidumpExceptionWriter();
  ~MinidumpExceptionWriter() override;

  //! \brief Initializes the MINIDUMP_EXCEPTION_STREAM based on \a
  //!     exception_snapshot.
  //!
  //! \param[in] exception_snapshot The exception snapshot to use as source
  //!     data.
  //! \param[in] thread_id_map A MinidumpThreadIDMap to be consulted to
  //!     determine the 32-bit minidump thread ID to use for the thread
  //!     identified by \a exception_snapshot.
  //!
  //! \note Valid in #kStateMutable. No mutator methods may be called before
  //!     this method, and it is not normally necessary to call any mutator
  //!     methods after this method.
  void InitializeFromSnapshot(const ExceptionSnapshot* exception_snapshot,
                              const MinidumpThreadIDMap& thread_id_map);

  //! \brief Arranges for MINIDUMP_EXCEPTION_STREAM::ThreadContext to point to
  //!     the CPU context to be written by \a context.
  //!
  //! A context is required in all MINIDUMP_EXCEPTION_STREAM objects.
  //!
  //! This object takes ownership of \a context and becomes its parent in the
  //! overall tree of internal::MinidumpWritable objects.
  //!
  //! \note Valid in #kStateMutable.
  void SetContext(std::unique_ptr<MinidumpContextWriter> context);

  //! \brief Sets MINIDUMP_EXCEPTION_STREAM::ThreadId.
  void SetThreadID(uint32_t thread_id) { exception_.ThreadId = thread_id; }

  //! \brief Sets MINIDUMP_EXCEPTION::ExceptionCode.
  void SetExceptionCode(uint32_t exception_code) {
    exception_.ExceptionRecord.ExceptionCode = exception_code;
  }

  //! \brief Sets MINIDUMP_EXCEPTION::ExceptionFlags.
  void SetExceptionFlags(uint32_t exception_flags) {
    exception_.ExceptionRecord.ExceptionFlags = exception_flags;
  }

  //! \brief Sets MINIDUMP_EXCEPTION::ExceptionRecord.
  void SetExceptionRecord(uint64_t exception_record) {
    exception_.ExceptionRecord.ExceptionRecord = exception_record;
  }

  //! \brief Sets MINIDUMP_EXCEPTION::ExceptionAddress.
  void SetExceptionAddress(uint64_t exception_address) {
    exception_.ExceptionRecord.ExceptionAddress = exception_address;
  }

  //! \brief Sets MINIDUMP_EXCEPTION::ExceptionInformation and
  //!     MINIDUMP_EXCEPTION::NumberParameters.
  //!
  //! MINIDUMP_EXCEPTION::NumberParameters is set to the number of elements in
  //! \a exception_information. The elements of
  //! MINIDUMP_EXCEPTION::ExceptionInformation are set to the elements of \a
  //! exception_information. Unused elements in
  //! MINIDUMP_EXCEPTION::ExceptionInformation are set to `0`.
  //!
  //! \a exception_information must have no more than
  //! #EXCEPTION_MAXIMUM_PARAMETERS elements.
  //!
  //! \note Valid in #kStateMutable.
  void SetExceptionInformation(
      const std::vector<uint64_t>& exception_information);

 protected:
  // MinidumpWritable:
  bool Freeze() override;
  size_t SizeOfObject() override;
  std::vector<MinidumpWritable*> Children() override;
  bool WriteObject(FileWriterInterface* file_writer) override;

  // MinidumpStreamWriter:
  MinidumpStreamType StreamType() const override;

 private:
  MINIDUMP_EXCEPTION_STREAM exception_;
  std::unique_ptr<MinidumpContextWriter> context_;

  DISALLOW_COPY_AND_ASSIGN(MinidumpExceptionWriter);
};

}  // namespace crashpad

#endif  // CRASHPAD_MINIDUMP_MINIDUMP_EXCEPTION_WRITER_H_
