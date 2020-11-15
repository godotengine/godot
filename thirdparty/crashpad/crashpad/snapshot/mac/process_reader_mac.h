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

#ifndef CRASHPAD_SNAPSHOT_MAC_PROCESS_READER_MAC_H_
#define CRASHPAD_SNAPSHOT_MAC_PROCESS_READER_MAC_H_

#include <mach/mach.h>
#include <stdint.h>
#include <sys/time.h>
#include <sys/types.h>
#include <time.h>

#include <memory>
#include <string>
#include <vector>

#include "base/macros.h"
#include "build/build_config.h"
#include "util/mach/task_memory.h"
#include "util/misc/initialization_state_dcheck.h"
#include "util/posix/process_info.h"

namespace crashpad {

class MachOImageReader;

//! \brief Accesses information about another process, identified by a Mach
//!     task.
class ProcessReaderMac {
 public:
  //! \brief Contains information about a thread that belongs to a task
  //!     (process).
  struct Thread {
#if defined(ARCH_CPU_X86_FAMILY)
    union ThreadContext {
      x86_thread_state64_t t64;
      x86_thread_state32_t t32;
    };
    union FloatContext {
      x86_float_state64_t f64;
      x86_float_state32_t f32;
    };
    union DebugContext {
      x86_debug_state64_t d64;
      x86_debug_state32_t d32;
    };
#endif

    Thread();
    ~Thread() {}

    ThreadContext thread_context;
    FloatContext float_context;
    DebugContext debug_context;
    uint64_t id;
    mach_vm_address_t stack_region_address;
    mach_vm_size_t stack_region_size;
    mach_vm_address_t thread_specific_data_address;
    thread_t port;
    int suspend_count;
    int priority;
  };

  //! \brief Contains information about a module loaded into a process.
  struct Module {
    Module();
    ~Module();

    //! \brief The pathname used to load the module from disk.
    std::string name;

    //! \brief An image reader for the module.
    //!
    //! The lifetime of this MachOImageReader is scoped to the lifetime of the
    //! ProcessReaderMac that created it.
    //!
    //! This field may be `nullptr` if a reader could not be created for the
    //! module.
    const MachOImageReader* reader;

    //! \brief The module’s timestamp.
    //!
    //! This field will be `0` if its value cannot be determined. It can only be
    //! determined for images that are loaded by dyld, so it will be `0` for the
    //! main executable and for dyld itself.
    time_t timestamp;
  };

  ProcessReaderMac();
  ~ProcessReaderMac();

  //! \brief Initializes this object. This method must be called before any
  //!     other.
  //!
  //! \param[in] task A send right to the target task’s task port. This object
  //!     does not take ownership of the send right.
  //!
  //! \return `true` on success, indicating that this object will respond
  //!     validly to further method calls. `false` on failure. On failure, no
  //!     further method calls should be made.
  bool Initialize(task_t task);

  //! \return `true` if the target task is a 64-bit process.
  bool Is64Bit() const { return is_64_bit_; }

  //! \return The target task’s process ID.
  pid_t ProcessID() const { return process_info_.ProcessID(); }

  //! \return The target task’s parent process ID.
  pid_t ParentProcessID() const { return process_info_.ParentProcessID(); }

  //! \brief Determines the target process’ start time.
  //!
  //! \param[out] start_time The time that the process started.
  void StartTime(timeval* start_time) const;

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

  //! \return Accesses the memory of the target task.
  TaskMemory* Memory() { return task_memory_.get(); }

  //! \return The threads that are in the task (process). The first element (at
  //!     index `0`) corresponds to the main thread.
  const std::vector<Thread>& Threads();

  //! \return The modules loaded in the process. The first element (at index
  //!     `0`) corresponds to the main executable, and the final element
  //!     corresponds to the dynamic loader, dyld.
  const std::vector<Module>& Modules();

  //! \brief Determines the location of the `dyld_all_image_infos` structure in
  //!     the process’ address space.
  //!
  //! This function is an internal implementation detail of Modules(), and
  //! should not normally be used directly. It is exposed solely for use by test
  //! code.
  //!
  //! \param[out] all_image_info_size The size of the `dyld_all_image_infos`
  //!     structure. Optional, may be `nullptr` if not required.
  //!
  //! \return The address of the `dyld_all_image_infos` structure in the
  //!     process’ address space, with \a all_image_info_size set appropriately.
  //!     On failure, returns `0` with a message logged.
  mach_vm_address_t DyldAllImageInfo(mach_vm_size_t* all_image_info_size);

 private:
  //! Performs lazy initialization of the \a threads_ vector on behalf of
  //! Threads().
  void InitializeThreads();

  //! Performs lazy initialization of the \a modules_ vector on behalf of
  //! Modules().
  void InitializeModules();

  //! \brief Calculates the base address and size of the region used as a
  //!     thread’s stack.
  //!
  //! The region returned by this method may be formed by merging multiple
  //! adjacent regions in a process’ memory map if appropriate. The base address
  //! of the returned region may be lower than the \a stack_pointer passed in
  //! when the ABI mandates a red zone below the stack pointer.
  //!
  //! \param[in] stack_pointer The stack pointer, referring to the top (lowest
  //!     address) of a thread’s stack.
  //! \param[out] stack_region_size The size of the memory region used as the
  //!     thread’s stack.
  //!
  //! \return The base address (lowest address) of the memory region used as the
  //!     thread’s stack.
  mach_vm_address_t CalculateStackRegion(mach_vm_address_t stack_pointer,
                                         mach_vm_size_t* stack_region_size);

  //! \brief Adjusts the region for the red zone, if the ABI requires one.
  //!
  //! This method performs red zone calculation for CalculateStackRegion(). Its
  //! parameters are local variables used within that method, and may be
  //! modified as needed.
  //!
  //! Where a red zone is required, the region of memory captured for a thread’s
  //! stack will be extended to include the red zone below the stack pointer,
  //! provided that such memory is mapped, readable, and has the correct user
  //! tag value. If these conditions cannot be met fully, as much of the red
  //! zone will be captured as is possible while meeting these conditions.
  //!
  //! \param[in,out] start_address The base address of the region to begin
  //!     capturing stack memory from. On entry, \a start_address is the stack
  //!     pointer. On return, \a start_address may be decreased to encompass a
  //!     red zone.
  //! \param[in,out] region_base The base address of the region that contains
  //!     stack memory. This is distinct from \a start_address in that \a
  //!     region_base will be page-aligned. On entry, \a region_base is the
  //!     base address of a region that contains \a start_address. On return,
  //!     if \a start_address is decremented and is outside of the region
  //!     originally described by \a region_base, \a region_base will also be
  //!     decremented appropriately.
  //! \param[in,out] region_size The size of the region that contains stack
  //!     memory. This region begins at \a region_base. On return, if \a
  //!     region_base is decremented, \a region_size will be incremented
  //!     appropriately.
  //! \param[in] user_tag The Mach VM system’s user tag for the region described
  //!     by the initial values of \a region_base and \a region_size. The red
  //!     zone will only be allowed to extend out of the region described by
  //!     these initial values if the user tag is appropriate for stack memory
  //!     and the expanded region has the same user tag value.
  void LocateRedZone(mach_vm_address_t* start_address,
                     mach_vm_address_t* region_base,
                     mach_vm_address_t* region_size,
                     unsigned int user_tag);

  ProcessInfo process_info_;
  std::vector<Thread> threads_;  // owns send rights
  std::vector<Module> modules_;
  std::vector<std::unique_ptr<MachOImageReader>> module_readers_;
  std::unique_ptr<TaskMemory> task_memory_;
  task_t task_;  // weak
  InitializationStateDcheck initialized_;

  // This shadows a method of process_info_, but it’s accessed so frequently
  // that it’s given a first-class field to save a call and a few bit operations
  // on each access.
  bool is_64_bit_;

  bool initialized_threads_;
  bool initialized_modules_;

  DISALLOW_COPY_AND_ASSIGN(ProcessReaderMac);
};

}  // namespace crashpad

#endif  // CRASHPAD_SNAPSHOT_MAC_PROCESS_READER_MAC_H_
