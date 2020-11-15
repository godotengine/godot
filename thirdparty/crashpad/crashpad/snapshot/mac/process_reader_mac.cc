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

#include "snapshot/mac/process_reader_mac.h"

#include <AvailabilityMacros.h>
#include <mach-o/loader.h>
#include <mach/mach_vm.h>

#include <algorithm>
#include <utility>

#include "base/logging.h"
#include "base/mac/mach_logging.h"
#include "base/mac/scoped_mach_port.h"
#include "base/mac/scoped_mach_vm.h"
#include "base/strings/stringprintf.h"
#include "snapshot/mac/mach_o_image_reader.h"
#include "snapshot/mac/process_types.h"
#include "util/misc/scoped_forbid_return.h"

namespace {

void MachTimeValueToTimeval(const time_value& mach, timeval* tv) {
  tv->tv_sec = mach.seconds;
  tv->tv_usec = mach.microseconds;
}

kern_return_t MachVMRegionRecurseDeepest(task_t task,
                                         mach_vm_address_t* address,
                                         mach_vm_size_t* size,
                                         natural_t* depth,
                                         vm_prot_t* protection,
                                         unsigned int* user_tag) {
  vm_region_submap_short_info_64 submap_info;
  mach_msg_type_number_t count = VM_REGION_SUBMAP_SHORT_INFO_COUNT_64;
  while (true) {
    kern_return_t kr = mach_vm_region_recurse(
        task,
        address,
        size,
        depth,
        reinterpret_cast<vm_region_recurse_info_t>(&submap_info),
        &count);
    if (kr != KERN_SUCCESS) {
      return kr;
    }

    if (!submap_info.is_submap) {
      *protection = submap_info.protection;
      *user_tag = submap_info.user_tag;
      return KERN_SUCCESS;
    }

    ++*depth;
  }
}

}  // namespace

namespace crashpad {

ProcessReaderMac::Thread::Thread()
    : thread_context(),
      float_context(),
      debug_context(),
      id(0),
      stack_region_address(0),
      stack_region_size(0),
      thread_specific_data_address(0),
      port(THREAD_NULL),
      suspend_count(0),
      priority(0) {}

ProcessReaderMac::Module::Module() : name(), reader(nullptr), timestamp(0) {}

ProcessReaderMac::Module::~Module() {}

ProcessReaderMac::ProcessReaderMac()
    : process_info_(),
      threads_(),
      modules_(),
      module_readers_(),
      task_memory_(),
      task_(TASK_NULL),
      initialized_(),
      is_64_bit_(false),
      initialized_threads_(false),
      initialized_modules_(false) {}

ProcessReaderMac::~ProcessReaderMac() {
  for (const Thread& thread : threads_) {
    kern_return_t kr = mach_port_deallocate(mach_task_self(), thread.port);
    MACH_LOG_IF(ERROR, kr != KERN_SUCCESS, kr) << "mach_port_deallocate";
  }
}

bool ProcessReaderMac::Initialize(task_t task) {
  INITIALIZATION_STATE_SET_INITIALIZING(initialized_);

  if (!process_info_.InitializeWithTask(task)) {
    return false;
  }

  is_64_bit_ = process_info_.Is64Bit();

  task_memory_.reset(new TaskMemory(task));
  task_ = task;

  INITIALIZATION_STATE_SET_VALID(initialized_);
  return true;
}

void ProcessReaderMac::StartTime(timeval* start_time) const {
  bool rv = process_info_.StartTime(start_time);
  DCHECK(rv);
}

bool ProcessReaderMac::CPUTimes(timeval* user_time,
                                timeval* system_time) const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);

  // Calculate user and system time the same way the kernel does for
  // getrusage(). See 10.9.2 xnu-2422.90.20/bsd/kern/kern_resource.c calcru().
  timerclear(user_time);
  timerclear(system_time);

  // As of the 10.8 SDK, the preferred routine is MACH_TASK_BASIC_INFO.
  // TASK_BASIC_INFO_64 is equivalent and works on earlier systems.
  task_basic_info_64 task_basic_info;
  mach_msg_type_number_t task_basic_info_count = TASK_BASIC_INFO_64_COUNT;
  kern_return_t kr = task_info(task_,
                               TASK_BASIC_INFO_64,
                               reinterpret_cast<task_info_t>(&task_basic_info),
                               &task_basic_info_count);
  if (kr != KERN_SUCCESS) {
    MACH_LOG(WARNING, kr) << "task_info TASK_BASIC_INFO_64";
    return false;
  }

  task_thread_times_info_data_t task_thread_times;
  mach_msg_type_number_t task_thread_times_count = TASK_THREAD_TIMES_INFO_COUNT;
  kr = task_info(task_,
                 TASK_THREAD_TIMES_INFO,
                 reinterpret_cast<task_info_t>(&task_thread_times),
                 &task_thread_times_count);
  if (kr != KERN_SUCCESS) {
    MACH_LOG(WARNING, kr) << "task_info TASK_THREAD_TIMES";
    return false;
  }

  MachTimeValueToTimeval(task_basic_info.user_time, user_time);
  MachTimeValueToTimeval(task_basic_info.system_time, system_time);

  timeval thread_user_time;
  MachTimeValueToTimeval(task_thread_times.user_time, &thread_user_time);
  timeval thread_system_time;
  MachTimeValueToTimeval(task_thread_times.system_time, &thread_system_time);

  timeradd(user_time, &thread_user_time, user_time);
  timeradd(system_time, &thread_system_time, system_time);

  return true;
}

const std::vector<ProcessReaderMac::Thread>& ProcessReaderMac::Threads() {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);

  if (!initialized_threads_) {
    InitializeThreads();
  }

  return threads_;
}

const std::vector<ProcessReaderMac::Module>& ProcessReaderMac::Modules() {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);

  if (!initialized_modules_) {
    InitializeModules();
  }

  return modules_;
}

mach_vm_address_t ProcessReaderMac::DyldAllImageInfo(
    mach_vm_size_t* all_image_info_size) {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);

  task_dyld_info_data_t dyld_info;
  mach_msg_type_number_t count = TASK_DYLD_INFO_COUNT;
  kern_return_t kr = task_info(
      task_, TASK_DYLD_INFO, reinterpret_cast<task_info_t>(&dyld_info), &count);
  if (kr != KERN_SUCCESS) {
    MACH_LOG(WARNING, kr) << "task_info";
    return 0;
  }

// TODO(mark): Deal with statically linked executables which don’t use dyld.
// This may look for the module that matches the executable path in the same
// data set that vmmap uses.

#if MAC_OS_X_VERSION_MAX_ALLOWED >= MAC_OS_X_VERSION_10_7
  // The task_dyld_info_data_t struct grew in 10.7, adding the format field.
  // Don’t check this field if it’s not present, which can happen when either
  // the SDK used at compile time or the kernel at run time are too old and
  // don’t know about it.
  if (count >= TASK_DYLD_INFO_COUNT) {
    const integer_t kExpectedFormat =
        !Is64Bit() ? TASK_DYLD_ALL_IMAGE_INFO_32 : TASK_DYLD_ALL_IMAGE_INFO_64;
    if (dyld_info.all_image_info_format != kExpectedFormat) {
      LOG(WARNING) << "unexpected task_dyld_info_data_t::all_image_info_format "
                   << dyld_info.all_image_info_format;
      DCHECK_EQ(dyld_info.all_image_info_format, kExpectedFormat);
      return 0;
    }
  }
#endif

  if (all_image_info_size) {
    *all_image_info_size = dyld_info.all_image_info_size;
  }
  return dyld_info.all_image_info_addr;
}

void ProcessReaderMac::InitializeThreads() {
  DCHECK(!initialized_threads_);
  DCHECK(threads_.empty());

  initialized_threads_ = true;

  thread_act_array_t threads;
  mach_msg_type_number_t thread_count = 0;
  kern_return_t kr = task_threads(task_, &threads, &thread_count);
  if (kr != KERN_SUCCESS) {
    MACH_LOG(WARNING, kr) << "task_threads";
    return;
  }

  // The send rights in the |threads| array won’t have their send rights managed
  // by anything until they’re added to |threads_| by the loop below. Any early
  // return (or exception) that happens between here and the completion of the
  // loop below will leak thread port send rights.
  ScopedForbidReturn threads_need_owners;

  base::mac::ScopedMachVM threads_vm(
      reinterpret_cast<vm_address_t>(threads),
      mach_vm_round_page(thread_count * sizeof(*threads)));

  for (size_t index = 0; index < thread_count; ++index) {
    Thread thread;
    thread.port = threads[index];

#if defined(ARCH_CPU_X86_FAMILY)
    const thread_state_flavor_t kThreadStateFlavor =
        Is64Bit() ? x86_THREAD_STATE64 : x86_THREAD_STATE32;
    mach_msg_type_number_t thread_state_count =
        Is64Bit() ? x86_THREAD_STATE64_COUNT : x86_THREAD_STATE32_COUNT;

    // TODO(mark): Use the AVX variants instead of the FLOAT variants?
    const thread_state_flavor_t kFloatStateFlavor =
        Is64Bit() ? x86_FLOAT_STATE64 : x86_FLOAT_STATE32;
    mach_msg_type_number_t float_state_count =
        Is64Bit() ? x86_FLOAT_STATE64_COUNT : x86_FLOAT_STATE32_COUNT;

    const thread_state_flavor_t kDebugStateFlavor =
        Is64Bit() ? x86_DEBUG_STATE64 : x86_DEBUG_STATE32;
    mach_msg_type_number_t debug_state_count =
        Is64Bit() ? x86_DEBUG_STATE64_COUNT : x86_DEBUG_STATE32_COUNT;
#endif

    kr = thread_get_state(
        thread.port,
        kThreadStateFlavor,
        reinterpret_cast<thread_state_t>(&thread.thread_context),
        &thread_state_count);
    if (kr != KERN_SUCCESS) {
      MACH_LOG(ERROR, kr) << "thread_get_state(" << kThreadStateFlavor << ")";
      continue;
    }

    kr = thread_get_state(
        thread.port,
        kFloatStateFlavor,
        reinterpret_cast<thread_state_t>(&thread.float_context),
        &float_state_count);
    if (kr != KERN_SUCCESS) {
      MACH_LOG(ERROR, kr) << "thread_get_state(" << kFloatStateFlavor << ")";
      continue;
    }

    kr = thread_get_state(
        thread.port,
        kDebugStateFlavor,
        reinterpret_cast<thread_state_t>(&thread.debug_context),
        &debug_state_count);
    if (kr != KERN_SUCCESS) {
      MACH_LOG(ERROR, kr) << "thread_get_state(" << kDebugStateFlavor << ")";
      continue;
    }

    thread_basic_info basic_info;
    mach_msg_type_number_t count = THREAD_BASIC_INFO_COUNT;
    kr = thread_info(thread.port,
                     THREAD_BASIC_INFO,
                     reinterpret_cast<thread_info_t>(&basic_info),
                     &count);
    if (kr != KERN_SUCCESS) {
      MACH_LOG(WARNING, kr) << "thread_info(THREAD_BASIC_INFO)";
    } else {
      thread.suspend_count = basic_info.suspend_count;
    }

    thread_identifier_info identifier_info;
    count = THREAD_IDENTIFIER_INFO_COUNT;
    kr = thread_info(thread.port,
                     THREAD_IDENTIFIER_INFO,
                     reinterpret_cast<thread_info_t>(&identifier_info),
                     &count);
    if (kr != KERN_SUCCESS) {
      MACH_LOG(WARNING, kr) << "thread_info(THREAD_IDENTIFIER_INFO)";
    } else {
      thread.id = identifier_info.thread_id;

      // thread_identifier_info::thread_handle contains the base of the
      // thread-specific data area, which on x86 and x86_64 is the thread’s base
      // address of the %gs segment. 10.9.2 xnu-2422.90.20/osfmk/kern/thread.c
      // thread_info_internal() gets the value from
      // machine_thread::cthread_self, which is the same value used to set the
      // %gs base in xnu-2422.90.20/osfmk/i386/pcb_native.c
      // act_machine_switch_pcb().
      //
      // This address is the internal pthread’s _pthread::tsd[], an array of
      // void* values that can be indexed by pthread_key_t values.
      thread.thread_specific_data_address = identifier_info.thread_handle;
    }

    thread_precedence_policy precedence;
    count = THREAD_PRECEDENCE_POLICY_COUNT;
    boolean_t get_default = FALSE;
    kr = thread_policy_get(thread.port,
                           THREAD_PRECEDENCE_POLICY,
                           reinterpret_cast<thread_policy_t>(&precedence),
                           &count,
                           &get_default);
    if (kr != KERN_SUCCESS) {
      MACH_LOG(INFO, kr) << "thread_policy_get";
    } else {
      thread.priority = precedence.importance;
    }

#if defined(ARCH_CPU_X86_FAMILY)
    mach_vm_address_t stack_pointer = Is64Bit()
                                          ? thread.thread_context.t64.__rsp
                                          : thread.thread_context.t32.__esp;
#endif

    thread.stack_region_address =
        CalculateStackRegion(stack_pointer, &thread.stack_region_size);

    threads_.push_back(thread);
  }

  threads_need_owners.Disarm();
}

void ProcessReaderMac::InitializeModules() {
  DCHECK(!initialized_modules_);
  DCHECK(modules_.empty());

  initialized_modules_ = true;

  mach_vm_size_t all_image_info_size;
  mach_vm_address_t all_image_info_addr =
      DyldAllImageInfo(&all_image_info_size);

  process_types::dyld_all_image_infos all_image_infos;
  if (!all_image_infos.Read(this, all_image_info_addr)) {
    LOG(WARNING) << "could not read dyld_all_image_infos";
    return;
  }

  if (all_image_infos.version < 1) {
    LOG(WARNING) << "unexpected dyld_all_image_infos version "
                 << all_image_infos.version;
    return;
  }

  size_t expected_size =
      process_types::dyld_all_image_infos::ExpectedSizeForVersion(
          this, all_image_infos.version);
  if (all_image_info_size < expected_size) {
    LOG(WARNING) << "small dyld_all_image_infos size " << all_image_info_size
                 << " < " << expected_size << " for version "
                 << all_image_infos.version;
    return;
  }

  // Note that all_image_infos.infoArrayCount may be 0 if a crash occurred while
  // dyld was loading the executable. This can happen if a required dynamic
  // library was not found. Similarly, all_image_infos.infoArray may be nullptr
  // if a crash occurred while dyld was updating it.
  //
  // TODO(mark): It may be possible to recover from these situations by looking
  // through memory mappings for Mach-O images.
  //
  // Continue along when this situation is detected, because even without any
  // images in infoArray, dyldImageLoadAddress may be set, and it may be
  // possible to recover some information from dyld.
  if (all_image_infos.infoArrayCount == 0) {
    LOG(WARNING) << "all_image_infos.infoArrayCount is zero";
  } else if (!all_image_infos.infoArray) {
    LOG(WARNING) << "all_image_infos.infoArray is nullptr";
  }

  std::vector<process_types::dyld_image_info> image_info_vector(
      all_image_infos.infoArrayCount);
  if (!process_types::dyld_image_info::ReadArrayInto(this,
                                                     all_image_infos.infoArray,
                                                     image_info_vector.size(),
                                                     &image_info_vector[0])) {
    LOG(WARNING) << "could not read dyld_image_info array";
    return;
  }

  size_t main_executable_count = 0;
  bool found_dyld = false;
  modules_.reserve(image_info_vector.size());
  for (const process_types::dyld_image_info& image_info : image_info_vector) {
    Module module;
    module.timestamp = image_info.imageFileModDate;

    if (!task_memory_->ReadCString(image_info.imageFilePath, &module.name)) {
      LOG(WARNING) << "could not read dyld_image_info::imageFilePath";
      // Proceed anyway with an empty module name.
    }

    std::unique_ptr<MachOImageReader> reader(new MachOImageReader());
    if (!reader->Initialize(this, image_info.imageLoadAddress, module.name)) {
      reader.reset();
    }

    module.reader = reader.get();

    uint32_t file_type = reader ? reader->FileType() : 0;

    module_readers_.push_back(std::move(reader));
    modules_.push_back(module);

    if (all_image_infos.version >= 2 && all_image_infos.dyldImageLoadAddress &&
        image_info.imageLoadAddress == all_image_infos.dyldImageLoadAddress) {
      found_dyld = true;
      LOG(WARNING) << base::StringPrintf(
          "found dylinker (%s) in dyld_all_image_infos::infoArray",
          module.name.c_str());

      LOG_IF(WARNING, file_type != MH_DYLINKER)
          << base::StringPrintf("dylinker (%s) has unexpected Mach-O type %d",
                                module.name.c_str(),
                                file_type);
    }

    if (file_type == MH_EXECUTE) {
      // On Mac OS X 10.6, the main executable does not normally show up at
      // index 0. This is because of how 10.6.8 dyld-132.13/src/dyld.cpp
      // notifyGDB(), the function resposible for causing
      // dyld_all_image_infos::infoArray to be updated, is called. It is
      // registered to be called when all dependents of an image have been
      // mapped (dyld_image_state_dependents_mapped), meaning that the main
      // executable won’t be added to the list until all of the libraries it
      // depends on are, even though dyld begins looking at the main executable
      // first. This changed in later versions of dyld, including those present
      // in 10.7. 10.9.4 dyld-239.4/src/dyld.cpp updateAllImages() (renamed from
      // notifyGDB()) is registered to be called when an image itself has been
      // mapped (dyld_image_state_mapped), regardless of the libraries that it
      // depends on.
      //
      // The interface requires that the main executable be first in the list,
      // so swap it into the right position.
      size_t index = modules_.size() - 1;
      if (main_executable_count == 0) {
        std::swap(modules_[0], modules_[index]);
      } else {
        LOG(WARNING) << base::StringPrintf(
            "multiple MH_EXECUTE modules (%s, %s)",
            modules_[0].name.c_str(),
            modules_[index].name.c_str());
      }
      ++main_executable_count;
    }
  }

  LOG_IF(WARNING, main_executable_count == 0) << "no MH_EXECUTE modules";

  // all_image_infos.infoArray doesn’t include an entry for dyld, but dyld is
  // loaded into the process’ address space as a module. Its load address is
  // easily known given a sufficiently recent all_image_infos.version, but the
  // timestamp and pathname are not given as they are for other modules.
  //
  // The timestamp is a lost cause, because the kernel doesn’t record the
  // timestamp of the dynamic linker at the time it’s loaded in the same way
  // that dyld records the timestamps of other modules when they’re loaded. (The
  // timestamp for the main executable is also not reported and appears as 0
  // even when accessed via dyld APIs, because it’s loaded by the kernel, not by
  // dyld.)
  //
  // The name can be determined, but it’s not as simple as hardcoding the
  // default "/usr/lib/dyld" because an executable could have specified anything
  // in its LC_LOAD_DYLINKER command.
  if (!found_dyld && all_image_infos.version >= 2 &&
      all_image_infos.dyldImageLoadAddress) {
    Module module;
    module.timestamp = 0;

    // Examine the executable’s LC_LOAD_DYLINKER load command to find the path
    // used to load dyld.
    if (all_image_infos.infoArrayCount >= 1 && main_executable_count >= 1) {
      module.name = modules_[0].reader->DylinkerName();
    }
    std::string module_name = !module.name.empty() ? module.name : "(dyld)";

    std::unique_ptr<MachOImageReader> reader(new MachOImageReader());
    if (!reader->Initialize(
            this, all_image_infos.dyldImageLoadAddress, module_name)) {
      reader.reset();
    }

    module.reader = reader.get();

    uint32_t file_type = reader ? reader->FileType() : 0;

    LOG_IF(WARNING, file_type != MH_DYLINKER)
        << base::StringPrintf("dylinker (%s) has unexpected Mach-O type %d",
                              module.name.c_str(),
                              file_type);

    if (module.name.empty() && file_type == MH_DYLINKER) {
      // Look inside dyld directly to find its preferred path.
      module.name = reader->DylinkerName();
    }

    if (module.name.empty()) {
      module.name = "(dyld)";
    }

    // dyld is loaded in the process even if its path can’t be determined.
    module_readers_.push_back(std::move(reader));
    modules_.push_back(module);
  }
}

mach_vm_address_t ProcessReaderMac::CalculateStackRegion(
    mach_vm_address_t stack_pointer,
    mach_vm_size_t* stack_region_size) {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);

  // For pthreads, it may be possible to compute the stack region based on the
  // internal _pthread::stackaddr and _pthread::stacksize. The _pthread struct
  // for a thread can be located at TSD slot 0, or the known offsets of
  // stackaddr and stacksize from the TSD area could be used.
  mach_vm_address_t region_base = stack_pointer;
  mach_vm_size_t region_size;
  natural_t depth = 0;
  vm_prot_t protection;
  unsigned int user_tag;
  kern_return_t kr = MachVMRegionRecurseDeepest(
      task_, &region_base, &region_size, &depth, &protection, &user_tag);
  if (kr != KERN_SUCCESS) {
    MACH_LOG(INFO, kr) << "mach_vm_region_recurse";
    *stack_region_size = 0;
    return 0;
  }

  if (region_base > stack_pointer) {
    // There’s nothing mapped at the stack pointer’s address. Something may have
    // trashed the stack pointer. Note that this shouldn’t happen for a normal
    // stack guard region violation because the guard region is mapped but has
    // VM_PROT_NONE protection.
    *stack_region_size = 0;
    return 0;
  }

  mach_vm_address_t start_address = stack_pointer;

  if ((protection & VM_PROT_READ) == 0) {
    // If the region isn’t readable, the stack pointer probably points to the
    // guard region. Don’t include it as part of the stack, and don’t include
    // anything at any lower memory address. The code below may still possibly
    // find the real stack region at a memory address higher than this region.
    start_address = region_base + region_size;
  } else {
    // If the ABI requires a red zone, adjust the region to include it if
    // possible.
    LocateRedZone(&start_address, &region_base, &region_size, user_tag);

    // Regardless of whether the ABI requires a red zone, capture up to
    // kExtraCaptureSize additional bytes of stack, but only if present in the
    // region that was already found.
    constexpr mach_vm_size_t kExtraCaptureSize = 128;
    start_address = std::max(start_address >= kExtraCaptureSize
                                 ? start_address - kExtraCaptureSize
                                 : start_address,
                             region_base);

    // Align start_address to a 16-byte boundary, which can help readers by
    // ensuring that data is aligned properly. This could page-align instead,
    // but that might be wasteful.
    constexpr mach_vm_size_t kDesiredAlignment = 16;
    start_address &= ~(kDesiredAlignment - 1);
    DCHECK_GE(start_address, region_base);
  }

  region_size -= (start_address - region_base);
  region_base = start_address;

  mach_vm_size_t total_region_size = region_size;

  // The stack region may have gotten split up into multiple abutting regions.
  // Try to coalesce them. This frequently happens for the main thread’s stack
  // when setrlimit(RLIMIT_STACK, …) is called. It may also happen if a region
  // is split up due to an mprotect() or vm_protect() call.
  //
  // Stack regions created by the kernel and the pthreads library will be marked
  // with the VM_MEMORY_STACK user tag. Scanning for multiple adjacent regions
  // with the same tag should find an entire stack region. Checking that the
  // protection on individual regions is not VM_PROT_NONE should guarantee that
  // this algorithm doesn’t collect map entries belonging to another thread’s
  // stack: well-behaved stacks (such as those created by the kernel and the
  // pthreads library) have VM_PROT_NONE guard regions at their low-address
  // ends.
  //
  // Other stack regions may not be so well-behaved and thus if user_tag is not
  // VM_MEMORY_STACK, the single region that was found is used as-is without
  // trying to merge it with other adjacent regions.
  if (user_tag == VM_MEMORY_STACK) {
    mach_vm_address_t try_address = region_base;
    mach_vm_address_t original_try_address;

    while (try_address += region_size,
           original_try_address = try_address,
           (kr = MachVMRegionRecurseDeepest(task_,
                                            &try_address,
                                            &region_size,
                                            &depth,
                                            &protection,
                                            &user_tag) == KERN_SUCCESS) &&
               try_address == original_try_address &&
               (protection & VM_PROT_READ) != 0 &&
               user_tag == VM_MEMORY_STACK) {
      total_region_size += region_size;
    }

    if (kr != KERN_SUCCESS && kr != KERN_INVALID_ADDRESS) {
      // Tolerate KERN_INVALID_ADDRESS because it will be returned when there
      // are no more regions in the map at or above the specified |try_address|.
      MACH_LOG(INFO, kr) << "mach_vm_region_recurse";
    }
  }

  *stack_region_size = total_region_size;
  return region_base;
}

void ProcessReaderMac::LocateRedZone(mach_vm_address_t* const start_address,
                                     mach_vm_address_t* const region_base,
                                     mach_vm_address_t* const region_size,
                                     const unsigned int user_tag) {
#if defined(ARCH_CPU_X86_FAMILY)
  if (Is64Bit()) {
    // x86_64 has a red zone. See AMD64 ABI 0.99.8,
    // https://raw.githubusercontent.com/wiki/hjl-tools/x86-psABI/x86-64-psABI-r252.pdf#page=19,
    // section 3.2.2, “The Stack Frame”.
    constexpr mach_vm_size_t kRedZoneSize = 128;
    mach_vm_address_t red_zone_base =
        *start_address >= kRedZoneSize ? *start_address - kRedZoneSize : 0;
    bool red_zone_ok = false;
    if (red_zone_base >= *region_base) {
      // The red zone is within the region already discovered.
      red_zone_ok = true;
    } else if (red_zone_base < *region_base && user_tag == VM_MEMORY_STACK) {
      // Probe to see if there’s a region immediately below the one already
      // discovered.
      mach_vm_address_t red_zone_region_base = red_zone_base;
      mach_vm_size_t red_zone_region_size;
      natural_t red_zone_depth = 0;
      vm_prot_t red_zone_protection;
      unsigned int red_zone_user_tag;
      kern_return_t kr = MachVMRegionRecurseDeepest(task_,
                                                    &red_zone_region_base,
                                                    &red_zone_region_size,
                                                    &red_zone_depth,
                                                    &red_zone_protection,
                                                    &red_zone_user_tag);
      if (kr != KERN_SUCCESS) {
        MACH_LOG(INFO, kr) << "mach_vm_region_recurse";
        *start_address = *region_base;
      } else if (red_zone_region_base + red_zone_region_size == *region_base &&
                 (red_zone_protection & VM_PROT_READ) != 0 &&
                 red_zone_user_tag == user_tag) {
        // The region containing the red zone is immediately below the region
        // already found, it’s readable (not the guard region), and it has the
        // same user tag as the region already found, so merge them.
        red_zone_ok = true;
        *region_base -= red_zone_region_size;
        *region_size += red_zone_region_size;
      }
    }

    if (red_zone_ok) {
      // Begin capturing from the base of the red zone (but not the entire
      // region that encompasses the red zone).
      *start_address = red_zone_base;
    } else {
      // The red zone would go lower into another region in memory, but no
      // region was found. Memory can only be captured to an address as low as
      // the base address of the region already found.
      *start_address = *region_base;
    }
  }
#endif
}

}  // namespace crashpad
