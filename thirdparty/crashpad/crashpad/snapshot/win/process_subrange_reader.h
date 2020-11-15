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

#ifndef CRASHPAD_SNAPSHOT_WIN_PROCESS_SUBRANGE_READER_WIN_H_
#define CRASHPAD_SNAPSHOT_WIN_PROCESS_SUBRANGE_READER_WIN_H_

#include <string>

#include "base/macros.h"
#include "util/misc/initialization_state_dcheck.h"
#include "util/win/address_types.h"
#include "util/win/checked_win_address_range.h"

namespace crashpad {

class ProcessReaderWin;

//! \brief A wrapper for ProcessReaderWin that only allows a specific subrange
//!     to be read from.
//!
//! This class is useful to restrict reads to a specific address range, such as
//! the address range occupied by a loaded module, or a specific section within
//! a module.
class ProcessSubrangeReader {
 public:
  ProcessSubrangeReader();
  ~ProcessSubrangeReader();

  //! \brief Initializes the object.
  //!
  //! \param[in] process_reader A reader for a remote process.
  //! \param[in] base The base address for the range that reads should be
  //!     restricted to.
  //! \param[in] size The size of the range that reads should be restricted to.
  //! \param[in] name The range’s name, a string to be used in logged messages.
  //!     This string is for diagnostic purposes.
  //!
  //! \return `true` on success, `false` on failure with a message logged. The
  //!     other methods in this class must not be called unless this method or
  //!     InitializeSubrange() has returned true.
  bool Initialize(ProcessReaderWin* process_reader,
                  WinVMAddress base,
                  WinVMSize size,
                  const std::string& name);

  //! \brief Initializes the object to a subrange of an existing
  //!     ProcessSubrangeReader.
  //!
  //! The subrange identified by \a base and \a size must be contained within
  //! the subrange in \a that.
  //!
  //! \param[in] that The existing ProcessSubrangeReader to base the new object
  //!     on.
  //! \param[in] base The base address for the range that reads should be
  //!     restricted to.
  //! \param[in] size The size of the range that reads should be restricted to.
  //! \param[in] sub_name A description of the subrange, which will be appended
  //!     to the \a name in \a that and used in logged messages. This string is
  //!     for diagnostic purposes.
  //!
  //! \return `true` on success, `false` on failure with a message logged. The
  //!     other methods in this class must not be called unless this method or
  //!     Initialize() has returned true.
  bool InitializeSubrange(const ProcessSubrangeReader& that,
                          WinVMAddress base,
                          WinVMSize size,
                          const std::string& sub_name);

  bool Is64Bit() const { return range_.Is64Bit(); }
  WinVMAddress Base() const { return range_.Base(); }
  WinVMAddress Size() const { return range_.Size(); }
  const std::string& name() const { return name_; }

  //! \brief Reads memory from the remote process.
  //!
  //! The range specified by \a address and \a size must be contained within
  //! the range that this object is permitted to read.
  //!
  //! \param[in] address The address to read from.
  //! \param[in] size The size of data to read, in bytes.
  //! \param[out] into The buffer to read data into.
  //!
  //! \return `true` on success, `false` on failure with a message logged.
  bool ReadMemory(WinVMAddress address, WinVMSize size, void* into) const;

 private:
  // Common helper for Initialize() and InitializeSubrange().
  bool InitializeInternal(ProcessReaderWin* process_reader,
                          WinVMAddress base,
                          WinVMSize size,
                          const std::string& name);

  std::string name_;
  CheckedWinAddressRange range_;
  ProcessReaderWin* process_reader_;  // weak
  InitializationStateDcheck initialized_;

  DISALLOW_COPY_AND_ASSIGN(ProcessSubrangeReader);
};

}  // namespace crashpad

#endif  // CRASHPAD_SNAPSHOT_WIN_PROCESS_SUBRANGE_READER_WIN_H_
