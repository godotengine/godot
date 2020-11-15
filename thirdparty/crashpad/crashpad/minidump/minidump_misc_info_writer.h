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

#ifndef CRASHPAD_MINIDUMP_MINIDUMP_MISC_INFO_WRITER_H_
#define CRASHPAD_MINIDUMP_MINIDUMP_MISC_INFO_WRITER_H_

#include <windows.h>
#include <dbghelp.h>
#include <stdint.h>
#include <sys/types.h>
#include <time.h>

#include <string>

#include "base/macros.h"
#include "minidump/minidump_stream_writer.h"
#include "minidump/minidump_writable.h"

namespace crashpad {

class ProcessSnapshot;

namespace internal {

//! \brief Returns the string to set in MINIDUMP_MISC_INFO_4::DbgBldStr.
//!
//! dbghelp produces strings like `"dbghelp.i386,6.3.9600.16520"` and
//! `"dbghelp.amd64,6.3.9600.16520"`. This function mimics that format, and adds
//! the OS that wrote the minidump along with any relevant platform-specific
//! data describing the compilation environment.
//!
//! This function is an implementation detail of
//! MinidumpMiscInfoWriter::InitializeFromSnapshot() and is only exposed for
//! testing purposes.
std::string MinidumpMiscInfoDebugBuildString();

}  // namespace internal

//! \brief The writer for a stream in the MINIDUMP_MISC_INFO family in a
//!     minidump file.
//!
//! The actual stream written will be a MINIDUMP_MISC_INFO,
//! MINIDUMP_MISC_INFO_2, MINIDUMP_MISC_INFO_3, MINIDUMP_MISC_INFO_4, or
//! MINIDUMP_MISC_INFO_5 stream. Later versions of MINIDUMP_MISC_INFO are
//! supersets of earlier versions. The earliest version that supports all of the
//! information that an object of this class contains will be used.
class MinidumpMiscInfoWriter final : public internal::MinidumpStreamWriter {
 public:
  MinidumpMiscInfoWriter();
  ~MinidumpMiscInfoWriter() override;

  //! \brief Initializes MINIDUMP_MISC_INFO_N based on \a process_snapshot.
  //!
  //! \param[in] process_snapshot The process snapshot to use as source data.
  //!
  //! \note Valid in #kStateMutable. No mutator methods may be called before
  //!     this method, and it is not normally necessary to call any mutator
  //!     methods after this method.
  void InitializeFromSnapshot(const ProcessSnapshot* process_snapshot);

  //! \brief Sets the field referenced by #MINIDUMP_MISC1_PROCESS_ID.
  void SetProcessID(uint32_t process_id);

  //! \brief Sets the fields referenced by #MINIDUMP_MISC1_PROCESS_TIMES.
  void SetProcessTimes(time_t process_create_time,
                       uint32_t process_user_time,
                       uint32_t process_kernel_time);

  //! \brief Sets the fields referenced by #MINIDUMP_MISC1_PROCESSOR_POWER_INFO.
  void SetProcessorPowerInfo(uint32_t processor_max_mhz,
                             uint32_t processor_current_mhz,
                             uint32_t processor_mhz_limit,
                             uint32_t processor_max_idle_state,
                             uint32_t processor_current_idle_state);

  //! \brief Sets the field referenced by #MINIDUMP_MISC3_PROCESS_INTEGRITY.
  void SetProcessIntegrityLevel(uint32_t process_integrity_level);

  //! \brief Sets the field referenced by #MINIDUMP_MISC3_PROCESS_EXECUTE_FLAGS.
  void SetProcessExecuteFlags(uint32_t process_execute_flags);

  //! \brief Sets the field referenced by #MINIDUMP_MISC3_PROTECTED_PROCESS.
  void SetProtectedProcess(uint32_t protected_process);

  //! \brief Sets the fields referenced by #MINIDUMP_MISC3_TIMEZONE.
  void SetTimeZone(uint32_t time_zone_id,
                   int32_t bias,
                   const std::string& standard_name,
                   const SYSTEMTIME& standard_date,
                   int32_t standard_bias,
                   const std::string& daylight_name,
                   const SYSTEMTIME& daylight_date,
                   int32_t daylight_bias);

  //! \brief Sets the fields referenced by #MINIDUMP_MISC4_BUILDSTRING.
  void SetBuildString(const std::string& build_string,
                      const std::string& debug_build_string);

  // TODO(mark): Provide a better interface than this. Don’t force callers to
  // build their own XSTATE_CONFIG_FEATURE_MSC_INFO structure.
  //
  //! \brief Sets MINIDUMP_MISC_INFO_5::XStateData.
  void SetXStateData(const XSTATE_CONFIG_FEATURE_MSC_INFO& xstate_data);

  //! \brief Sets the field referenced by #MINIDUMP_MISC5_PROCESS_COOKIE.
  void SetProcessCookie(uint32_t process_cookie);

 protected:
  // MinidumpWritable:
  bool Freeze() override;
  size_t SizeOfObject() override;
  bool WriteObject(FileWriterInterface* file_writer) override;
  MinidumpStreamType StreamType() const override;

 private:
  //! \brief Returns the size of the object to be written based on
  //!     MINIDUMP_MISC_INFO_N::Flags1.
  //!
  //! The smallest defined structure type in the MINIDUMP_MISC_INFO family that
  //! can hold all of the data that has been populated will be used.
  size_t CalculateSizeOfObjectFromFlags() const;

  MINIDUMP_MISC_INFO_N misc_info_;
  bool has_xstate_data_;

  DISALLOW_COPY_AND_ASSIGN(MinidumpMiscInfoWriter);
};

}  // namespace crashpad

#endif  // CRASHPAD_MINIDUMP_MINIDUMP_MISC_INFO_WRITER_H_
