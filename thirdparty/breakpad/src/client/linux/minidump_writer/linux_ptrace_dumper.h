// Copyright (c) 2012, Google Inc.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// linux_ptrace_dumper.h: Define the google_breakpad::LinuxPtraceDumper
// class, which is derived from google_breakpad::LinuxDumper to extract
// information from a crashed process via ptrace.
// This class was originally splitted from google_breakpad::LinuxDumper.

#ifndef CLIENT_LINUX_MINIDUMP_WRITER_LINUX_PTRACE_DUMPER_H_
#define CLIENT_LINUX_MINIDUMP_WRITER_LINUX_PTRACE_DUMPER_H_

#include "client/linux/minidump_writer/linux_dumper.h"

namespace google_breakpad {

class LinuxPtraceDumper : public LinuxDumper {
 public:
  // Constructs a dumper for extracting information of a given process
  // with a process ID of |pid|.
  explicit LinuxPtraceDumper(pid_t pid);

  // Implements LinuxDumper::BuildProcPath().
  // Builds a proc path for a certain pid for a node (/proc/<pid>/<node>).
  // |path| is a character array of at least NAME_MAX bytes to return the
  // result. |node| is the final node without any slashes. Returns true on
  // success.
  virtual bool BuildProcPath(char* path, pid_t pid, const char* node) const;

  // Implements LinuxDumper::CopyFromProcess().
  // Copies content of |length| bytes from a given process |child|,
  // starting from |src|, into |dest|. This method uses ptrace to extract
  // the content from the target process. Always returns true.
  virtual bool CopyFromProcess(void* dest, pid_t child, const void* src,
                               size_t length);

  // Implements LinuxDumper::GetThreadInfoByIndex().
  // Reads information about the |index|-th thread of |threads_|.
  // Returns true on success. One must have called |ThreadsSuspend| first.
  virtual bool GetThreadInfoByIndex(size_t index, ThreadInfo* info);

  // Implements LinuxDumper::IsPostMortem().
  // Always returns false to indicate this dumper performs a dump of
  // a crashed process via ptrace.
  virtual bool IsPostMortem() const;

  // Implements LinuxDumper::ThreadsSuspend().
  // Suspends all threads in the given process. Returns true on success.
  virtual bool ThreadsSuspend();

  // Implements LinuxDumper::ThreadsResume().
  // Resumes all threads in the given process. Returns true on success.
  virtual bool ThreadsResume();

 protected:
  // Implements LinuxDumper::EnumerateThreads().
  // Enumerates all threads of the given process into |threads_|.
  virtual bool EnumerateThreads();

 private:
  // Set to true if all threads of the crashed process are suspended.
  bool threads_suspended_;

  // Read the tracee's registers on kernel with PTRACE_GETREGSET support.
  // Returns false if PTRACE_GETREGSET is not defined.
  // Returns true on success.
  bool ReadRegisterSet(ThreadInfo* info, pid_t tid);

  // Read the tracee's registers on kernel with PTRACE_GETREGS support.
  // Returns true on success.
  bool ReadRegisters(ThreadInfo* info, pid_t tid);
};

}  // namespace google_breakpad

#endif  // CLIENT_LINUX_HANDLER_LINUX_PTRACE_DUMPER_H_
