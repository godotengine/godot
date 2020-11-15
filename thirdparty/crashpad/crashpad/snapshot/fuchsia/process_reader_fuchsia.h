// Copyright 2018 The Crashpad Authors. All rights reserved.
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

#ifndef CRASHPAD_SNAPSHOT_FUCHSIA_PROCESS_READER_H_
#define CRASHPAD_SNAPSHOT_FUCHSIA_PROCESS_READER_H_

#include <lib/zx/process.h>
#include <zircon/syscalls/debug.h>

#include <memory>
#include <vector>

#include "base/macros.h"
#include "build/build_config.h"
#include "snapshot/elf/elf_image_reader.h"
#include "snapshot/fuchsia/memory_map_fuchsia.h"
#include "snapshot/module_snapshot.h"
#include "util/misc/initialization_state_dcheck.h"
#include "util/numeric/checked_range.h"
#include "util/process/process_memory_fuchsia.h"
#include "util/process/process_memory_range.h"

namespace crashpad {

//! \brief Accesses information about another process, identified by a Fuchsia
//!     process.
class ProcessReaderFuchsia {
 public:
  //! \brief Contains information about a module loaded into a process.
  struct Module {
    Module();
    ~Module();

    //! \brief The `ZX_PROP_NAME` of the module.
    std::string name;

    //! \brief An image reader for the module.
    //!
    //! The lifetime of this ElfImageReader is scoped to the lifetime of the
    //! ProcessReaderFuchsia that created it.
    //!
    //! This field may be `nullptr` if a reader could not be created for the
    //! module.
    ElfImageReader* reader;

    //! \brief The module's type.
    ModuleSnapshot::ModuleType type = ModuleSnapshot::kModuleTypeUnknown;
  };

  //! \brief Contains information about a thread that belongs to a process.
  struct Thread {
    Thread();
    ~Thread();

    //! \brief The kernel identifier for the thread.
    zx_koid_t id = ZX_KOID_INVALID;

    //! \brief The state of the thread, the `ZX_THREAD_STATE_*` value or `-1` if
    //!     the value could not be retrieved.
    uint32_t state = -1;

    //! \brief The `ZX_PROP_NAME` property of the thread. This may be empty.
    std::string name;

    //! \brief The raw architecture-specific `zx_thread_state_general_regs_t` as
    //!     returned by `zx_thread_read_state()`.
    zx_thread_state_general_regs_t general_registers = {};

    //! \brief The regions representing the stack. The first entry in the vector
    //!     represents the callstack, and further entries optionally identify
    //!     other stack data when the thread uses a split stack representation.
    std::vector<CheckedRange<zx_vaddr_t, size_t>> stack_regions;
  };

  ProcessReaderFuchsia();
  ~ProcessReaderFuchsia();

  //! \brief Initializes this object. This method must be called before any
  //!     other.
  //!
  //! \param[in] process A process handle with permissions to read properties
  //!     and memory from the target process.
  //!
  //! \return `true` on success, indicating that this object will respond
  //!     validly to further method calls. `false` on failure. On failure, no
  //!     further method calls should be made.
  bool Initialize(const zx::process& process);

  //! \return The modules loaded in the process. The first element (at index
  //!     `0`) corresponds to the main executable.
  const std::vector<Module>& Modules();

  //! \return The threads that are in the process.
  const std::vector<Thread>& Threads();

  //! \brief Return a memory reader for the target process.
  ProcessMemory* Memory() { return process_memory_.get(); }

 private:
  //! Performs lazy initialization of the \a modules_ vector on behalf of
  //! Modules().
  void InitializeModules();

  //! Performs lazy initialization of the \a threads_ vector on behalf of
  //! Threads().
  void InitializeThreads();

  std::vector<Module> modules_;
  std::vector<Thread> threads_;
  std::vector<std::unique_ptr<ElfImageReader>> module_readers_;
  std::vector<std::unique_ptr<ProcessMemoryRange>> process_memory_ranges_;
  std::unique_ptr<ProcessMemoryFuchsia> process_memory_;
  MemoryMapFuchsia memory_map_;
  zx::unowned_process process_;
  bool initialized_modules_ = false;
  bool initialized_threads_ = false;
  InitializationStateDcheck initialized_;

  DISALLOW_COPY_AND_ASSIGN(ProcessReaderFuchsia);
};

}  // namespace crashpad

#endif  // CRASHPAD_SNAPSHOT_FUCHSIA_PROCESS_READER_H_
