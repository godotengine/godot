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

#ifndef CRASHPAD_SNAPSHOT_LINUX_PROCESS_READER_LINUX_H_
#define CRASHPAD_SNAPSHOT_LINUX_PROCESS_READER_LINUX_H_

#include <sys/time.h>
#include <sys/types.h>

#include <memory>
#include <string>
#include <vector>

#include "base/macros.h"
#include "snapshot/elf/elf_image_reader.h"
#include "snapshot/module_snapshot.h"
#include "util/linux/address_types.h"
#include "util/linux/memory_map.h"
#include "util/linux/ptrace_connection.h"
#include "util/linux/thread_info.h"
#include "util/misc/initialization_state_dcheck.h"
#include "util/posix/process_info.h"
#include "util/process/process_memory.h"

namespace crashpad {

//! \brief Accesses information about another process, identified by a process
//!     ID.
class ProcessReaderLinux {
 public:
  //! \brief Contains information about a thread that belongs to a process.
  struct Thread {
    Thread();
    ~Thread();

    //! \brief Initializes the thread's stack using \a stack_pointer instead of
    //!   the stack pointer in \a thread_info.
    //!
    //! This method initializes \a stack_region_address and \a stack_region_size
    //! overwriting any values they previously contained. This is useful, for
    //! example, if the thread is currently in a signal handler context, which
    //! may execute on a different stack than was used before the signal was
    //! received.
    //!
    //! \param[in] reader A process reader for the target process.
    //! \param[in] stack_pointer The stack pointer for the stack to initialize.
    void InitializeStackFromSP(ProcessReaderLinux* reader,
                               LinuxVMAddress stack_pointer);

    ThreadInfo thread_info;
    LinuxVMAddress stack_region_address;
    LinuxVMSize stack_region_size;
    pid_t tid;
    int sched_policy;
    int static_priority;
    int nice_value;

    //! \brief `true` if `sched_policy`, `static_priority`, and `nice_value` are
    //!     all valid.
    bool have_priorities;

   private:
    friend class ProcessReaderLinux;

    bool InitializePtrace(PtraceConnection* connection);
    void InitializeStack(ProcessReaderLinux* reader);
  };

  //! \brief Contains information about a module loaded into a process.
  struct Module {
    Module();
    ~Module();

    //! \brief The pathname used to load the module from disk.
    std::string name;

    //! \brief An image reader for the module.
    //!
    //! The lifetime of this ElfImageReader is scoped to the lifetime of the
    //! ProcessReaderLinux that created it.
    //!
    //! This field may be `nullptr` if a reader could not be created for the
    //! module.
    ElfImageReader* elf_reader;

    //! \brief The module's type.
    ModuleSnapshot::ModuleType type;
  };

  ProcessReaderLinux();
  ~ProcessReaderLinux();

  //! \brief Initializes this object.
  //!
  //! This method must be successfully called before calling any other method in
  //! this class and may only be called once.
  //!
  //! \param[in] connection A PtraceConnection to the target process.
  //! \return `true` on success. `false` on failure with a message logged.
  bool Initialize(PtraceConnection* connection);

  //! \brief Return `true` if the target task is a 64-bit process.
  bool Is64Bit() const { return is_64_bit_; }

  //! \brief Return the target process' process ID.
  pid_t ProcessID() const { return process_info_.ProcessID(); }

  //! \brief Return the target process' parent process ID.
  pid_t ParentProcessID() const { return process_info_.ParentProcessID(); }

  //! \brief Return a memory reader for the target process.
  ProcessMemory* Memory() { return connection_->Memory(); }

  //! \brief Return a memory map of the target process.
  MemoryMap* GetMemoryMap() { return &memory_map_; }

  //! \brief Determines the target process’ start time.
  //!
  //! \param[out] start_time The time that the process started.
  //! \return `true` on success with \a start_time set. Otherwise `false` with a
  //!     message logged.
  bool StartTime(timeval* start_time) const;

  //! \brief Determines the target process’ execution time.
  //!
  //! \param[out] user_time The amount of time the process has executed code in
  //!     user mode.
  //! \param[out] system_time The amount of time the process has executed code
  //!     in system mode.
  //!
  //! \return `true` on success, `false` on failure, with a warning logged. On
  //!     failure, \a user_time and \a system_time will be set to represent no
  //!     time spent executing code in user or system mode.
  bool CPUTimes(timeval* user_time, timeval* system_time) const;

  //! \brief Return a vector of threads that are in the task process. If the
  //!     main thread is able to be identified and traced, it will be placed at
  //!     index `0`.
  const std::vector<Thread>& Threads();

  //! \return The modules loaded in the process. The first element (at index
  //!     `0`) corresponds to the main executable.
  const std::vector<Module>& Modules();

 private:
  void InitializeThreads();
  void InitializeModules();

  PtraceConnection* connection_;  // weak
  ProcessInfo process_info_;
  MemoryMap memory_map_;
  std::vector<Thread> threads_;
  std::vector<Module> modules_;
  std::vector<std::unique_ptr<ElfImageReader>> elf_readers_;
  bool is_64_bit_;
  bool initialized_threads_;
  bool initialized_modules_;
  InitializationStateDcheck initialized_;

  DISALLOW_COPY_AND_ASSIGN(ProcessReaderLinux);
};

}  // namespace crashpad

#endif  // CRASHPAD_SNAPSHOT_LINUX_PROCESS_READER_LINUX_H_
