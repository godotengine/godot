// Copyright 2012 Google LLC
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
//     * Neither the name of Google LLC nor the names of its
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

// linux_core_dumper.h: Define the google_breakpad::LinuxCoreDumper
// class, which is derived from google_breakpad::LinuxDumper to extract
// information from a crashed process via its core dump and proc files.

#ifndef CLIENT_LINUX_MINIDUMP_WRITER_LINUX_CORE_DUMPER_H_
#define CLIENT_LINUX_MINIDUMP_WRITER_LINUX_CORE_DUMPER_H_

#include "client/linux/minidump_writer/linux_dumper.h"
#include "common/linux/elf_core_dump.h"
#include "common/linux/memory_mapped_file.h"

namespace google_breakpad {

class LinuxCoreDumper : public LinuxDumper {
 public:
  // Constructs a dumper for extracting information of a given process
  // with a process ID of |pid| via its core dump file at |core_path| and
  // its proc files at |procfs_path|. If |procfs_path| is a copy of
  // /proc/<pid>, it should contain the following files:
  //     auxv, cmdline, environ, exe, maps, status
  // See LinuxDumper for the purpose of |root_prefix|.
  LinuxCoreDumper(pid_t pid, const char* core_path, const char* procfs_path,
                  const char* root_prefix = "");

  // Implements LinuxDumper::BuildProcPath().
  // Builds a proc path for a certain pid for a node (/proc/<pid>/<node>).
  // |path| is a character array of at least NAME_MAX bytes to return the
  // result.|node| is the final node without any slashes. Return true on
  // success.
  //
  // As this dumper performs a post-mortem dump and makes use of a copy
  // of the proc files of the crashed process, this derived method does
  // not actually make use of |pid| and always returns a subpath of
  // |procfs_path_| regardless of whether |pid| corresponds to the main
  // process or a thread of the process, i.e. assuming both the main process
  // and its threads have the following proc files with the same content:
  //     auxv, cmdline, environ, exe, maps, status
  virtual bool BuildProcPath(char* path, pid_t pid, const char* node) const;

  // Implements LinuxDumper::CopyFromProcess().
  // Copies content of |length| bytes from a given process |child|,
  // starting from |src|, into |dest|. This method extracts the content
  // the core dump and fills |dest| with a sequence of marker bytes
  // if the expected data is not found in the core dump. Returns true if
  // the expected data is found in the core dump.
  virtual bool CopyFromProcess(void* dest, pid_t child, const void* src,
                               size_t length);

  // Implements LinuxDumper::GetThreadInfoByIndex().
  // Reads information about the |index|-th thread of |threads_|.
  // Returns true on success. One must have called |ThreadsSuspend| first.
  virtual bool GetThreadInfoByIndex(size_t index, ThreadInfo* info);

  // Implements LinuxDumper::IsPostMortem().
  // Always returns true to indicate that this dumper performs a
  // post-mortem dump of a crashed process via a core dump file.
  virtual bool IsPostMortem() const;

  // Implements LinuxDumper::ThreadsSuspend().
  // As the dumper performs a post-mortem dump via a core dump file,
  // there is no threads to suspend. This method does nothing and
  // always returns true.
  virtual bool ThreadsSuspend();

  // Implements LinuxDumper::ThreadsResume().
  // As the dumper performs a post-mortem dump via a core dump file,
  // there is no threads to resume. This method does nothing and
  // always returns true.
  virtual bool ThreadsResume();

 protected:
  // Implements LinuxDumper::EnumerateThreads().
  // Enumerates all threads of the given process into |threads_|.
  virtual bool EnumerateThreads();

 private:
  // Path of the core dump file.
  const char* core_path_;

  // Path of the directory containing the proc files of the given process,
  // which is usually a copy of /proc/<pid>.
  const char* procfs_path_;

  // Memory-mapped core dump file at |core_path_|.
  MemoryMappedFile mapped_core_file_;

  // Content of the core dump file.
  ElfCoreDump core_;

  // Thread info found in the core dump file.
  wasteful_vector<ThreadInfo> thread_infos_;
};

}  // namespace google_breakpad

#endif  // CLIENT_LINUX_HANDLER_LINUX_CORE_DUMPER_H_
