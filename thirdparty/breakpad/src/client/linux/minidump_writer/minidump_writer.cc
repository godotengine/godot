// Copyright 2010 Google LLC
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

// This code writes out minidump files:
//   http://msdn.microsoft.com/en-us/library/ms680378(VS.85,loband).aspx
//
// Minidumps are a Microsoft format which Breakpad uses for recording crash
// dumps. This code has to run in a compromised environment (the address space
// may have received SIGSEGV), thus the following rules apply:
//   * You may not enter the dynamic linker. This means that we cannot call
//     any symbols in a shared library (inc libc). Because of this we replace
//     libc functions in linux_libc_support.h.
//   * You may not call syscalls via the libc wrappers. This rule is a subset
//     of the first rule but it bears repeating. We have direct wrappers
//     around the system calls in linux_syscall_support.h.
//   * You may not malloc. There's an alternative allocator in memory.h and
//     a canonical instance in the LinuxDumper object. We use the placement
//     new form to allocate objects and we don't delete them.

#ifdef HAVE_CONFIG_H
#include <config.h>  // Must come first
#endif

#include "client/linux/handler/minidump_descriptor.h"
#include "client/linux/minidump_writer/minidump_writer.h"
#include "client/minidump_file_writer-inl.h"

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <link.h>
#include <stdio.h>
#if defined(__ANDROID__)
#include <sys/system_properties.h>
#endif
#include <sys/types.h>
#include <sys/ucontext.h>
#include <sys/user.h>
#include <sys/utsname.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>

#include "client/linux/dump_writer_common/thread_info.h"
#include "client/linux/dump_writer_common/ucontext_reader.h"
#include "client/linux/handler/exception_handler.h"
#include "client/linux/minidump_writer/cpu_set.h"
#include "client/linux/minidump_writer/line_reader.h"
#include "client/linux/minidump_writer/linux_dumper.h"
#include "client/linux/minidump_writer/linux_ptrace_dumper.h"
#include "client/linux/minidump_writer/pe_file.h"
#include "client/linux/minidump_writer/pe_structs.h"
#include "client/linux/minidump_writer/proc_cpuinfo_reader.h"
#include "client/minidump_file_writer.h"
#include "common/linux/file_id.h"
#include "common/linux/linux_libc_support.h"
#include "common/minidump_type_helper.h"
#include "google_breakpad/common/minidump_format.h"
#include "third_party/lss/linux_syscall_support.h"

namespace {

using google_breakpad::AppMemoryList;
using google_breakpad::auto_wasteful_vector;
using google_breakpad::elf::kDefaultBuildIdSize;
using google_breakpad::ExceptionHandler;
using google_breakpad::CpuSet;
using google_breakpad::LineReader;
using google_breakpad::LinuxDumper;
using google_breakpad::LinuxPtraceDumper;
using google_breakpad::MDTypeHelper;
using google_breakpad::MappingEntry;
using google_breakpad::MappingInfo;
using google_breakpad::MappingList;
using google_breakpad::MinidumpFileWriter;
using google_breakpad::PageAllocator;
using google_breakpad::PEFile;
using google_breakpad::PEFileFormat;
using google_breakpad::ProcCpuInfoReader;
using google_breakpad::RawContextCPU;
using google_breakpad::RSDS_DEBUG_FORMAT;
using google_breakpad::ThreadInfo;
using google_breakpad::TypedMDRVA;
using google_breakpad::UContextReader;
using google_breakpad::UntypedMDRVA;
using google_breakpad::wasteful_vector;

typedef MDTypeHelper<sizeof(void*)>::MDRawDebug MDRawDebug;
typedef MDTypeHelper<sizeof(void*)>::MDRawLinkMap MDRawLinkMap;

class MinidumpWriter {
 public:
  // The following kLimit* constants are for when minidump_size_limit_ is set
  // and the minidump size might exceed it.
  //
  // Estimate for how big each thread's stack will be (in bytes).
  static const unsigned kLimitAverageThreadStackLength = 8 * 1024;
  // Number of threads whose stack size we don't want to limit.  These base
  // threads will simply be the first N threads returned by the dumper (although
  // the crashing thread will never be limited).  Threads beyond this count are
  // the extra threads.
  static const unsigned kLimitBaseThreadCount = 20;
  // Maximum stack size to dump for any extra thread (in bytes).
  static const unsigned kLimitMaxExtraThreadStackLen = 2 * 1024;
  // Make sure this number of additional bytes can fit in the minidump
  // (exclude the stack data).
  static const unsigned kLimitMinidumpFudgeFactor = 64 * 1024;

  MinidumpWriter(const char* minidump_path,
                 int minidump_fd,
                 const ExceptionHandler::CrashContext* context,
                 const MappingList& mappings,
                 const AppMemoryList& appmem,
                 bool skip_stacks_if_mapping_unreferenced,
                 uintptr_t principal_mapping_address,
                 bool sanitize_stacks,
                 LinuxDumper* dumper)
      : fd_(minidump_fd),
        path_(minidump_path),
        ucontext_(context ? &context->context : NULL),
#if GOOGLE_BREAKPAD_CRASH_CONTEXT_HAS_FLOAT_STATE
        float_state_(context ? &context->float_state : NULL),
#endif
        dumper_(dumper),
        minidump_size_limit_(-1),
        memory_blocks_(dumper_->allocator()),
        mapping_list_(mappings),
        app_memory_list_(appmem),
        skip_stacks_if_mapping_unreferenced_(
            skip_stacks_if_mapping_unreferenced),
        principal_mapping_address_(principal_mapping_address),
        principal_mapping_(nullptr),
    sanitize_stacks_(sanitize_stacks) {
    // Assert there should be either a valid fd or a valid path, not both.
    assert(fd_ != -1 || minidump_path);
    assert(fd_ == -1 || !minidump_path);
  }

  bool Init() {
    if (!dumper_->Init())
      return false;

    if (!dumper_->ThreadsSuspend() || !dumper_->LateInit())
      return false;

    if (skip_stacks_if_mapping_unreferenced_) {
      principal_mapping_ =
          dumper_->FindMappingNoBias(principal_mapping_address_);
      if (!CrashingThreadReferencesPrincipalMapping())
        return false;
    }

    if (fd_ != -1)
      minidump_writer_.SetFile(fd_);
    else if (!minidump_writer_.Open(path_))
      return false;

    return true;
  }

  ~MinidumpWriter() {
    // Don't close the file descriptor when it's been provided explicitly.
    // Callers might still need to use it.
    if (fd_ == -1)
      minidump_writer_.Close();
    dumper_->ThreadsResume();
  }

  bool CrashingThreadReferencesPrincipalMapping() {
    if (!ucontext_ || !principal_mapping_)
      return false;

    const uintptr_t low_addr =
        principal_mapping_->system_mapping_info.start_addr;
    const uintptr_t high_addr =
        principal_mapping_->system_mapping_info.end_addr;

    const uintptr_t stack_pointer = UContextReader::GetStackPointer(ucontext_);
    const uintptr_t pc = UContextReader::GetInstructionPointer(ucontext_);

    if (pc >= low_addr && pc < high_addr)
      return true;

    uint8_t* stack_copy;
    const void* stack;
    size_t stack_len;

    if (!dumper_->GetStackInfo(&stack, &stack_len, stack_pointer))
      return false;

    stack_copy = reinterpret_cast<uint8_t*>(Alloc(stack_len));
    dumper_->CopyFromProcess(stack_copy, GetCrashThread(), stack, stack_len);

    uintptr_t stack_pointer_offset =
        stack_pointer - reinterpret_cast<uintptr_t>(stack);

    return dumper_->StackHasPointerToMapping(
        stack_copy, stack_len, stack_pointer_offset, *principal_mapping_);
  }

  bool Dump() {
    // A minidump file contains a number of tagged streams. This is the number
    // of stream which we write.
    unsigned kNumWriters = 13;

    TypedMDRVA<MDRawDirectory> dir(&minidump_writer_);
    {
      // Ensure the header gets flushed, as that happens in the destructor.
      // If a crash occurs somewhere below, at least the header will be
      // intact.
      TypedMDRVA<MDRawHeader> header(&minidump_writer_);
      if (!header.Allocate())
        return false;

      if (!dir.AllocateArray(kNumWriters))
        return false;

      my_memset(header.get(), 0, sizeof(MDRawHeader));

      header.get()->signature = MD_HEADER_SIGNATURE;
      header.get()->version = MD_HEADER_VERSION;
      header.get()->time_date_stamp = time(NULL);
      header.get()->stream_count = kNumWriters;
      header.get()->stream_directory_rva = dir.position();
    }

    unsigned dir_index = 0;
    MDRawDirectory dirent;

    if (!WriteThreadListStream(&dirent))
      return false;
    dir.CopyIndex(dir_index++, &dirent);

    if (!WriteMappings(&dirent))
      return false;
    dir.CopyIndex(dir_index++, &dirent);

    if (!WriteAppMemory())
      return false;

    if (!WriteMemoryListStream(&dirent))
      return false;
    dir.CopyIndex(dir_index++, &dirent);

    if (!WriteExceptionStream(&dirent))
      return false;
    dir.CopyIndex(dir_index++, &dirent);

    if (!WriteSystemInfoStream(&dirent))
      return false;
    dir.CopyIndex(dir_index++, &dirent);

    dirent.stream_type = MD_LINUX_CPU_INFO;
    if (!WriteFile(&dirent.location, "/proc/cpuinfo"))
      NullifyDirectoryEntry(&dirent);
    dir.CopyIndex(dir_index++, &dirent);

    dirent.stream_type = MD_LINUX_PROC_STATUS;
    if (!WriteProcFile(&dirent.location, GetCrashThread(), "status"))
      NullifyDirectoryEntry(&dirent);
    dir.CopyIndex(dir_index++, &dirent);

    dirent.stream_type = MD_LINUX_LSB_RELEASE;
    if (!WriteFile(&dirent.location, "/etc/lsb-release"))
      NullifyDirectoryEntry(&dirent);
    dir.CopyIndex(dir_index++, &dirent);

    dirent.stream_type = MD_LINUX_CMD_LINE;
    if (!WriteProcFile(&dirent.location, GetCrashThread(), "cmdline"))
      NullifyDirectoryEntry(&dirent);
    dir.CopyIndex(dir_index++, &dirent);

    dirent.stream_type = MD_LINUX_ENVIRON;
    if (!WriteProcFile(&dirent.location, GetCrashThread(), "environ"))
      NullifyDirectoryEntry(&dirent);
    dir.CopyIndex(dir_index++, &dirent);

    dirent.stream_type = MD_LINUX_AUXV;
    if (!WriteProcFile(&dirent.location, GetCrashThread(), "auxv"))
      NullifyDirectoryEntry(&dirent);
    dir.CopyIndex(dir_index++, &dirent);

    dirent.stream_type = MD_LINUX_MAPS;
    if (!WriteProcFile(&dirent.location, GetCrashThread(), "maps"))
      NullifyDirectoryEntry(&dirent);
    dir.CopyIndex(dir_index++, &dirent);

    dirent.stream_type = MD_LINUX_DSO_DEBUG;
    if (!WriteDSODebugStream(&dirent))
      NullifyDirectoryEntry(&dirent);
    dir.CopyIndex(dir_index++, &dirent);

    // If you add more directory entries, don't forget to update kNumWriters,
    // above.

    dumper_->ThreadsResume();
    return true;
  }

  bool FillThreadStack(MDRawThread* thread, uintptr_t stack_pointer,
                       uintptr_t pc, int max_stack_len, uint8_t** stack_copy) {
    *stack_copy = NULL;
    const void* stack;
    size_t stack_len;

    thread->stack.start_of_memory_range = stack_pointer;
    thread->stack.memory.data_size = 0;
    thread->stack.memory.rva = minidump_writer_.position();

    if (dumper_->GetStackInfo(&stack, &stack_len, stack_pointer)) {
      if (max_stack_len >= 0 &&
          stack_len > static_cast<unsigned int>(max_stack_len)) {
        stack_len = max_stack_len;
        // Skip empty chunks of length max_stack_len.
        uintptr_t int_stack = reinterpret_cast<uintptr_t>(stack);
        if (max_stack_len > 0) {
          while (int_stack + max_stack_len < stack_pointer) {
            int_stack += max_stack_len;
          }
        }
        stack = reinterpret_cast<const void*>(int_stack);
      }
      *stack_copy = reinterpret_cast<uint8_t*>(Alloc(stack_len));
      dumper_->CopyFromProcess(*stack_copy, thread->thread_id, stack,
                               stack_len);

      uintptr_t stack_pointer_offset =
          stack_pointer - reinterpret_cast<uintptr_t>(stack);
      if (skip_stacks_if_mapping_unreferenced_) {
        if (!principal_mapping_) {
          return true;
        }
        uintptr_t low_addr = principal_mapping_->system_mapping_info.start_addr;
        uintptr_t high_addr = principal_mapping_->system_mapping_info.end_addr;
        if ((pc < low_addr || pc > high_addr) &&
            !dumper_->StackHasPointerToMapping(*stack_copy, stack_len,
                                               stack_pointer_offset,
                                               *principal_mapping_)) {
          return true;
        }
      }

      if (sanitize_stacks_) {
        dumper_->SanitizeStackCopy(*stack_copy, stack_len, stack_pointer,
                                   stack_pointer_offset);
      }

      UntypedMDRVA memory(&minidump_writer_);
      if (!memory.Allocate(stack_len))
        return false;
      memory.Copy(*stack_copy, stack_len);
      thread->stack.start_of_memory_range = reinterpret_cast<uintptr_t>(stack);
      thread->stack.memory = memory.location();
      memory_blocks_.push_back(thread->stack);
    }
    return true;
  }

  // Write information about the threads.
  bool WriteThreadListStream(MDRawDirectory* dirent) {
    const unsigned num_threads = dumper_->threads().size();

    TypedMDRVA<uint32_t> list(&minidump_writer_);
    if (!list.AllocateObjectAndArray(num_threads, sizeof(MDRawThread)))
      return false;

    dirent->stream_type = MD_THREAD_LIST_STREAM;
    dirent->location = list.location();

    *list.get() = num_threads;

    // If there's a minidump size limit, check if it might be exceeded.  Since
    // most of the space is filled with stack data, just check against that.
    // If this expects to exceed the limit, set extra_thread_stack_len such
    // that any thread beyond the first kLimitBaseThreadCount threads will
    // have only kLimitMaxExtraThreadStackLen bytes dumped.
    int extra_thread_stack_len = -1;  // default to no maximum
    if (minidump_size_limit_ >= 0) {
      const unsigned estimated_total_stack_size = num_threads *
          kLimitAverageThreadStackLength;
      const off_t estimated_minidump_size = minidump_writer_.position() +
          estimated_total_stack_size + kLimitMinidumpFudgeFactor;
      if (estimated_minidump_size > minidump_size_limit_)
        extra_thread_stack_len = kLimitMaxExtraThreadStackLen;
    }

    for (unsigned i = 0; i < num_threads; ++i) {
      MDRawThread thread;
      my_memset(&thread, 0, sizeof(thread));
      thread.thread_id = dumper_->threads()[i];

      // We have a different source of information for the crashing thread. If
      // we used the actual state of the thread we would find it running in the
      // signal handler with the alternative stack, which would be deeply
      // unhelpful.
      if (static_cast<pid_t>(thread.thread_id) == GetCrashThread() &&
          ucontext_ &&
          !dumper_->IsPostMortem()) {
        uint8_t* stack_copy;
        const uintptr_t stack_ptr = UContextReader::GetStackPointer(ucontext_);
        if (!FillThreadStack(&thread, stack_ptr,
                             UContextReader::GetInstructionPointer(ucontext_),
                             -1, &stack_copy))
          return false;

        // Copy 256 bytes around crashing instruction pointer to minidump.
        const size_t kIPMemorySize = 256;
        uint64_t ip = UContextReader::GetInstructionPointer(ucontext_);
        // Bound it to the upper and lower bounds of the memory map
        // it's contained within. If it's not in mapped memory,
        // don't bother trying to write it.
        bool ip_is_mapped = false;
        MDMemoryDescriptor ip_memory_d;
        for (unsigned j = 0; j < dumper_->mappings().size(); ++j) {
          const MappingInfo& mapping = *dumper_->mappings()[j];
          if (ip >= mapping.start_addr &&
              ip < mapping.start_addr + mapping.size) {
            ip_is_mapped = true;
            // Try to get 128 bytes before and after the IP, but
            // settle for whatever's available.
            ip_memory_d.start_of_memory_range =
              std::max(mapping.start_addr,
                       uintptr_t(ip - (kIPMemorySize / 2)));
            uintptr_t end_of_range =
              std::min(uintptr_t(ip + (kIPMemorySize / 2)),
                       uintptr_t(mapping.start_addr + mapping.size));
            ip_memory_d.memory.data_size =
              end_of_range - ip_memory_d.start_of_memory_range;
            break;
          }
        }

        if (ip_is_mapped) {
          UntypedMDRVA ip_memory(&minidump_writer_);
          if (!ip_memory.Allocate(ip_memory_d.memory.data_size))
            return false;
          uint8_t* memory_copy =
              reinterpret_cast<uint8_t*>(Alloc(ip_memory_d.memory.data_size));
          dumper_->CopyFromProcess(
              memory_copy,
              thread.thread_id,
              reinterpret_cast<void*>(ip_memory_d.start_of_memory_range),
              ip_memory_d.memory.data_size);
          ip_memory.Copy(memory_copy, ip_memory_d.memory.data_size);
          ip_memory_d.memory = ip_memory.location();
          memory_blocks_.push_back(ip_memory_d);
        }

        TypedMDRVA<RawContextCPU> cpu(&minidump_writer_);
        if (!cpu.Allocate())
          return false;
        my_memset(cpu.get(), 0, sizeof(RawContextCPU));
#if GOOGLE_BREAKPAD_CRASH_CONTEXT_HAS_FLOAT_STATE
        UContextReader::FillCPUContext(cpu.get(), ucontext_, float_state_);
#else
        UContextReader::FillCPUContext(cpu.get(), ucontext_);
#endif
        thread.thread_context = cpu.location();
        crashing_thread_context_ = cpu.location();
      } else {
        ThreadInfo info;
        if (!dumper_->GetThreadInfoByIndex(i, &info))
          return false;

        uint8_t* stack_copy;
        int max_stack_len = -1;  // default to no maximum for this thread
        if (minidump_size_limit_ >= 0 && i >= kLimitBaseThreadCount)
          max_stack_len = extra_thread_stack_len;
        if (!FillThreadStack(&thread, info.stack_pointer,
                             info.GetInstructionPointer(), max_stack_len,
                             &stack_copy))
          return false;

        TypedMDRVA<RawContextCPU> cpu(&minidump_writer_);
        if (!cpu.Allocate())
          return false;
        my_memset(cpu.get(), 0, sizeof(RawContextCPU));
        info.FillCPUContext(cpu.get());
        thread.thread_context = cpu.location();
        if (dumper_->threads()[i] == GetCrashThread()) {
          crashing_thread_context_ = cpu.location();
          if (!dumper_->IsPostMortem()) {
            // This is the crashing thread of a live process, but
            // no context was provided, so set the crash address
            // while the instruction pointer is already here.
            dumper_->set_crash_address(info.GetInstructionPointer());
          }
        }
      }

      list.CopyIndexAfterObject(i, &thread, sizeof(thread));
    }

    return true;
  }

  // Write application-provided memory regions.
  bool WriteAppMemory() {
    for (AppMemoryList::const_iterator iter = app_memory_list_.begin();
         iter != app_memory_list_.end();
         ++iter) {
      uint8_t* data_copy =
        reinterpret_cast<uint8_t*>(dumper_->allocator()->Alloc(iter->length));
      dumper_->CopyFromProcess(data_copy, GetCrashThread(), iter->ptr,
                               iter->length);

      UntypedMDRVA memory(&minidump_writer_);
      if (!memory.Allocate(iter->length)) {
        return false;
      }
      memory.Copy(data_copy, iter->length);
      MDMemoryDescriptor desc;
      desc.start_of_memory_range = reinterpret_cast<uintptr_t>(iter->ptr);
      desc.memory = memory.location();
      memory_blocks_.push_back(desc);
    }

    return true;
  }

  static bool ShouldIncludeMapping(const MappingInfo& mapping) {
    if (mapping.name[0] == 0 ||  // only want modules with filenames.
        // Only want to include one mapping per shared lib.
        // Avoid filtering executable mappings.
        (mapping.offset != 0 && !mapping.exec) ||
        mapping.size < 4096) {  // too small to get a signature for.
      return false;
    }

    return true;
  }

  // If there is caller-provided information about this mapping
  // in the mapping_list_ list, return true. Otherwise, return false.
  bool HaveMappingInfo(const MappingInfo& mapping) {
    for (MappingList::const_iterator iter = mapping_list_.begin();
         iter != mapping_list_.end();
         ++iter) {
      // Ignore any mappings that are wholly contained within
      // mappings in the mapping_info_ list.
      if (mapping.start_addr >= iter->first.start_addr &&
          (mapping.start_addr + mapping.size) <=
          (iter->first.start_addr + iter->first.size)) {
        return true;
      }
    }
    return false;
  }

  // Write information about the mappings in effect. Because we are using the
  // minidump format, the information about the mappings is pretty limited.
  // Because of this, we also include the full, unparsed, /proc/$x/maps file in
  // another stream in the file.
  bool WriteMappings(MDRawDirectory* dirent) {
    const unsigned num_mappings = dumper_->mappings().size();
    unsigned num_output_mappings = mapping_list_.size();

    for (unsigned i = 0; i < dumper_->mappings().size(); ++i) {
      const MappingInfo& mapping = *dumper_->mappings()[i];
      if (ShouldIncludeMapping(mapping) && !HaveMappingInfo(mapping))
        num_output_mappings++;
    }

    TypedMDRVA<uint32_t> list(&minidump_writer_);
    if (num_output_mappings) {
      if (!list.AllocateObjectAndArray(num_output_mappings, MD_MODULE_SIZE))
        return false;
    } else {
      // Still create the module list stream, although it will have zero
      // modules.
      if (!list.Allocate())
        return false;
    }

    dirent->stream_type = MD_MODULE_LIST_STREAM;
    dirent->location = list.location();
    *list.get() = num_output_mappings;

    // First write all the mappings from the dumper
    unsigned int j = 0;
    for (unsigned i = 0; i < num_mappings; ++i) {
      const MappingInfo& mapping = *dumper_->mappings()[i];
      if (!ShouldIncludeMapping(mapping) || HaveMappingInfo(mapping))
        continue;

      MDRawModule mod;
      if (!FillRawModule(mapping, true, i, &mod, NULL))
        return false;
      list.CopyIndexAfterObject(j++, &mod, MD_MODULE_SIZE);
    }
    // Next write all the mappings provided by the caller
    for (MappingList::const_iterator iter = mapping_list_.begin();
         iter != mapping_list_.end();
         ++iter) {
      MDRawModule mod;
      if (!FillRawModule(iter->first, false, 0, &mod, iter->second))
        return false;
      list.CopyIndexAfterObject(j++, &mod, MD_MODULE_SIZE);
    }

    return true;
  }

  // Fill the MDRawModule |mod| with information about the provided
  // |mapping|. If |identifier| is non-NULL, use it instead of calculating
  // a file ID from the mapping.
  bool FillRawModule(const MappingInfo& mapping,
                     bool member,
                     unsigned int mapping_id,
                     MDRawModule* mod,
                     const uint8_t* identifier) {
    my_memset(mod, 0, MD_MODULE_SIZE);

    mod->base_of_image = mapping.start_addr;
    mod->size_of_image = mapping.size;

    char file_name[NAME_MAX];
    char file_path[NAME_MAX];

    dumper_->GetMappingEffectiveNameAndPath(mapping, file_path,
                                            sizeof(file_path), file_name,
                                            sizeof(file_name));

    RSDS_DEBUG_FORMAT rsds;
    PEFileFormat file_format = PEFile::TryGetDebugInfo(file_path, &rsds);

    if (file_format == PEFileFormat::notPeCoff) {
      // The module is not a PE/COFF file, process as an ELF.
      auto_wasteful_vector<uint8_t, kDefaultBuildIdSize> identifier_bytes(
          dumper_->allocator());

      if (identifier) {
        // GUID was provided by caller.
        identifier_bytes.insert(identifier_bytes.end(), identifier,
                                identifier + sizeof(MDGUID));
      } else {
        // Note: ElfFileIdentifierForMapping() can manipulate the
        // |mapping.name|, that is why we need to call the method
        // GetMappingEffectiveNameAndPath again.
        dumper_->ElfFileIdentifierForMapping(mapping, member, mapping_id,
                                             identifier_bytes);
        dumper_->GetMappingEffectiveNameAndPath(mapping, file_path,
                                                sizeof(file_path), file_name,
                                                sizeof(file_name));
      }

      if (!identifier_bytes.empty()) {
        UntypedMDRVA cv(&minidump_writer_);
        if (!cv.Allocate(MDCVInfoELF_minsize + identifier_bytes.size()))
          return false;

        const uint32_t cv_signature = MD_CVINFOELF_SIGNATURE;
        cv.Copy(&cv_signature, sizeof(cv_signature));
        cv.Copy(cv.position() + sizeof(cv_signature), &identifier_bytes[0],
                identifier_bytes.size());

        mod->cv_record = cv.location();
      }
    } else {
      // The module is a PE/COFF file. Create MDCVInfoPDB70 struct for it.
      size_t file_name_length = strlen(file_name);
      TypedMDRVA<MDCVInfoPDB70> cv(&minidump_writer_);
      if (!cv.AllocateObjectAndArray(file_name_length + 1, sizeof(uint8_t)))
        return false;
      if (!cv.CopyIndexAfterObject(0, file_name, file_name_length))
        return false;
      MDCVInfoPDB70* cv_ptr = cv.get();
      cv_ptr->cv_signature = MD_CVINFOPDB70_SIGNATURE;
      if (file_format == PEFileFormat::peWithBuildId) {
        // Populate BuildId and age using RSDS instance.
        cv_ptr->signature.data1 = static_cast<uint32_t>(rsds.guid[0]) << 24 |
                                  static_cast<uint32_t>(rsds.guid[1]) << 16 |
                                  static_cast<uint32_t>(rsds.guid[2]) << 8 |
                                  static_cast<uint32_t>(rsds.guid[3]);
        cv_ptr->signature.data2 =
            static_cast<uint16_t>(rsds.guid[4]) << 8 | rsds.guid[5];
        cv_ptr->signature.data3 =
            static_cast<uint16_t>(rsds.guid[6]) << 8 | rsds.guid[7];
        cv_ptr->signature.data4[0] = rsds.guid[8];
        cv_ptr->signature.data4[1] = rsds.guid[9];
        cv_ptr->signature.data4[2] = rsds.guid[10];
        cv_ptr->signature.data4[3] = rsds.guid[11];
        cv_ptr->signature.data4[4] = rsds.guid[12];
        cv_ptr->signature.data4[5] = rsds.guid[13];
        cv_ptr->signature.data4[6] = rsds.guid[14];
        cv_ptr->signature.data4[7] = rsds.guid[15];
        // The Age field should be reverted as well.
        cv_ptr->age = static_cast<uint32_t>(rsds.age[0]) << 24 |
                      static_cast<uint32_t>(rsds.age[1]) << 16 |
                      static_cast<uint32_t>(rsds.age[2]) << 8 |
                      static_cast<uint32_t>(rsds.age[3]);
      } else {
        cv_ptr->age = 0;
      }

      mod->cv_record = cv.location();
    }

    MDLocationDescriptor ld;
    if (!minidump_writer_.WriteString(file_path, my_strlen(file_path), &ld))
      return false;
    mod->module_name_rva = ld.rva;
    return true;
  }

  bool WriteMemoryListStream(MDRawDirectory* dirent) {
    TypedMDRVA<uint32_t> list(&minidump_writer_);
    if (memory_blocks_.size()) {
      if (!list.AllocateObjectAndArray(memory_blocks_.size(),
                                       sizeof(MDMemoryDescriptor)))
        return false;
    } else {
      // Still create the memory list stream, although it will have zero
      // memory blocks.
      if (!list.Allocate())
        return false;
    }

    dirent->stream_type = MD_MEMORY_LIST_STREAM;
    dirent->location = list.location();

    *list.get() = memory_blocks_.size();

    for (size_t i = 0; i < memory_blocks_.size(); ++i) {
      list.CopyIndexAfterObject(i, &memory_blocks_[i],
                                sizeof(MDMemoryDescriptor));
    }
    return true;
  }

  bool WriteExceptionStream(MDRawDirectory* dirent) {
    TypedMDRVA<MDRawExceptionStream> exc(&minidump_writer_);
    if (!exc.Allocate())
      return false;

    MDRawExceptionStream* stream = exc.get();
    my_memset(stream, 0, sizeof(MDRawExceptionStream));

    dirent->stream_type = MD_EXCEPTION_STREAM;
    dirent->location = exc.location();

    stream->thread_id = GetCrashThread();
    stream->exception_record.exception_code = dumper_->crash_signal();
    stream->exception_record.exception_flags = dumper_->crash_signal_code();
    stream->exception_record.exception_address = dumper_->crash_address();
    const std::vector<uint64_t> crash_exception_info =
        dumper_->crash_exception_info();
    stream->exception_record.number_parameters = crash_exception_info.size();
    memcpy(stream->exception_record.exception_information,
           crash_exception_info.data(),
           sizeof(uint64_t) * crash_exception_info.size());
    stream->thread_context = crashing_thread_context_;

    return true;
  }

  bool WriteSystemInfoStream(MDRawDirectory* dirent) {
    TypedMDRVA<MDRawSystemInfo> si(&minidump_writer_);
    if (!si.Allocate())
      return false;
    my_memset(si.get(), 0, sizeof(MDRawSystemInfo));

    dirent->stream_type = MD_SYSTEM_INFO_STREAM;
    dirent->location = si.location();

    WriteCPUInformation(si.get());
    WriteOSInformation(si.get());

    return true;
  }

  bool WriteDSODebugStream(MDRawDirectory* dirent) {
    ElfW(Phdr)* phdr = reinterpret_cast<ElfW(Phdr)*>(dumper_->auxv()[AT_PHDR]);
    char* base;
    int phnum = dumper_->auxv()[AT_PHNUM];
    if (!phnum || !phdr)
      return false;

    // Assume the program base is at the beginning of the same page as the PHDR
    base = reinterpret_cast<char*>(reinterpret_cast<uintptr_t>(phdr) & ~0xfff);

    // Search for the program PT_DYNAMIC segment
    ElfW(Addr) dyn_addr = 0;
    for (; phnum >= 0; phnum--, phdr++) {
      ElfW(Phdr) ph;
      if (!dumper_->CopyFromProcess(&ph, GetCrashThread(), phdr, sizeof(ph)))
        return false;

      // Adjust base address with the virtual address of the PT_LOAD segment
      // corresponding to offset 0
      if (ph.p_type == PT_LOAD && ph.p_offset == 0) {
        base -= ph.p_vaddr;
      }
      if (ph.p_type == PT_DYNAMIC) {
        dyn_addr = ph.p_vaddr;
      }
    }
    if (!dyn_addr)
      return false;

    ElfW(Dyn)* dynamic = reinterpret_cast<ElfW(Dyn)*>(dyn_addr + base);

    // The dynamic linker makes information available that helps gdb find all
    // DSOs loaded into the program. If this information is indeed available,
    // dump it to a MD_LINUX_DSO_DEBUG stream.
    struct r_debug* r_debug = NULL;
    uint32_t dynamic_length = 0;

    for (int i = 0; ; ++i) {
      ElfW(Dyn) dyn;
      dynamic_length += sizeof(dyn);
      if (!dumper_->CopyFromProcess(&dyn, GetCrashThread(), dynamic + i,
                                    sizeof(dyn))) {
        return false;
      }

#ifdef __mips__
      const int32_t debug_tag = DT_MIPS_RLD_MAP;
#else
      const int32_t debug_tag = DT_DEBUG;
#endif
      if (dyn.d_tag == debug_tag) {
        r_debug = reinterpret_cast<struct r_debug*>(dyn.d_un.d_ptr);
        continue;
      } else if (dyn.d_tag == DT_NULL) {
        break;
      }
    }

    // The "r_map" field of that r_debug struct contains a linked list of all
    // loaded DSOs.
    // Our list of DSOs potentially is different from the ones in the crashing
    // process. So, we have to be careful to never dereference pointers
    // directly. Instead, we use CopyFromProcess() everywhere.
    // See <link.h> for a more detailed discussion of the how the dynamic
    // loader communicates with debuggers.

    // Count the number of loaded DSOs
    int dso_count = 0;
    struct r_debug debug_entry;
    if (!dumper_->CopyFromProcess(&debug_entry, GetCrashThread(), r_debug,
                                  sizeof(debug_entry))) {
      return false;
    }
    for (struct link_map* ptr = debug_entry.r_map; ptr; ) {
      struct link_map map;
      if (!dumper_->CopyFromProcess(&map, GetCrashThread(), ptr, sizeof(map)))
        return false;

      ptr = map.l_next;
      dso_count++;
    }

    MDRVA linkmap_rva = minidump_writer_.kInvalidMDRVA;
    if (dso_count > 0) {
      // If we have at least one DSO, create an array of MDRawLinkMap
      // entries in the minidump file.
      TypedMDRVA<MDRawLinkMap> linkmap(&minidump_writer_);
      if (!linkmap.AllocateArray(dso_count))
        return false;
      linkmap_rva = linkmap.location().rva;
      int idx = 0;

      // Iterate over DSOs and write their information to mini dump
      for (struct link_map* ptr = debug_entry.r_map; ptr; ) {
        struct link_map map;
        if (!dumper_->CopyFromProcess(&map, GetCrashThread(), ptr, sizeof(map)))
          return  false;

        ptr = map.l_next;
        char filename[257] = { 0 };
        if (map.l_name) {
          dumper_->CopyFromProcess(filename, GetCrashThread(), map.l_name,
                                   sizeof(filename) - 1);
        }
        MDLocationDescriptor location;
        if (!minidump_writer_.WriteString(filename, 0, &location))
          return false;
        MDRawLinkMap entry;
        entry.name = location.rva;
        entry.addr = map.l_addr;
        entry.ld = reinterpret_cast<uintptr_t>(map.l_ld);
        linkmap.CopyIndex(idx++, &entry);
      }
    }

    // Write MD_LINUX_DSO_DEBUG record
    TypedMDRVA<MDRawDebug> debug(&minidump_writer_);
    if (!debug.AllocateObjectAndArray(1, dynamic_length))
      return false;
    my_memset(debug.get(), 0, sizeof(MDRawDebug));
    dirent->stream_type = MD_LINUX_DSO_DEBUG;
    dirent->location = debug.location();

    debug.get()->version = debug_entry.r_version;
    debug.get()->map = linkmap_rva;
    debug.get()->dso_count = dso_count;
    debug.get()->brk = debug_entry.r_brk;
    debug.get()->ldbase = debug_entry.r_ldbase;
    debug.get()->dynamic = reinterpret_cast<uintptr_t>(dynamic);

    wasteful_vector<char> dso_debug_data(dumper_->allocator(), dynamic_length);
    // The passed-in size to the constructor (above) is only a hint.
    // Must call .resize() to do actual initialization of the elements.
    dso_debug_data.resize(dynamic_length);
    dumper_->CopyFromProcess(&dso_debug_data[0], GetCrashThread(), dynamic,
                             dynamic_length);
    debug.CopyIndexAfterObject(0, &dso_debug_data[0], dynamic_length);

    return true;
  }

  void set_minidump_size_limit(off_t limit) { minidump_size_limit_ = limit; }

 private:
  void* Alloc(unsigned bytes) {
    return dumper_->allocator()->Alloc(bytes);
  }

  pid_t GetCrashThread() const {
    return dumper_->crash_thread();
  }

  void NullifyDirectoryEntry(MDRawDirectory* dirent) {
    dirent->stream_type = 0;
    dirent->location.data_size = 0;
    dirent->location.rva = 0;
  }

#if defined(__i386__) || defined(__x86_64__) || defined(__mips__)
  bool WriteCPUInformation(MDRawSystemInfo* sys_info) {
    char vendor_id[sizeof(sys_info->cpu.x86_cpu_info.vendor_id) + 1] = {0};
    static const char vendor_id_name[] = "vendor_id";

    struct CpuInfoEntry {
      const char* info_name;
      int value;
      bool found;
    } cpu_info_table[] = {
      { "processor", -1, false },
#if defined(__i386__) || defined(__x86_64__)
      { "model", 0, false },
      { "stepping",  0, false },
      { "cpu family", 0, false },
#endif
    };

    // processor_architecture should always be set, do this first
    sys_info->processor_architecture =
#if defined(__mips__)
# if _MIPS_SIM == _ABIO32
        MD_CPU_ARCHITECTURE_MIPS;
# elif _MIPS_SIM == _ABI64
        MD_CPU_ARCHITECTURE_MIPS64;
# else
#  error "This mips ABI is currently not supported (n32)"
#endif
#elif defined(__i386__)
        MD_CPU_ARCHITECTURE_X86;
#else
        MD_CPU_ARCHITECTURE_AMD64;
#endif

    const int fd = sys_open("/proc/cpuinfo", O_RDONLY, 0);
    if (fd < 0)
      return false;

    {
      PageAllocator allocator;
      ProcCpuInfoReader* const reader = new(allocator) ProcCpuInfoReader(fd);
      const char* field;
      while (reader->GetNextField(&field)) {
        bool is_first_entry = true;
        for (CpuInfoEntry& entry : cpu_info_table) {
          if (!is_first_entry && entry.found) {
            // except for the 'processor' field, ignore repeated values.
            continue;
          }
          is_first_entry = false;
          if (!my_strcmp(field, entry.info_name)) {
            size_t value_len;
            const char* value = reader->GetValueAndLen(&value_len);
            if (value_len == 0)
              continue;

            uintptr_t val;
            if (my_read_decimal_ptr(&val, value) == value)
              continue;

            entry.value = static_cast<int>(val);
            entry.found = true;
          }
        }

        // special case for vendor_id
        if (!my_strcmp(field, vendor_id_name)) {
          size_t value_len;
          const char* value = reader->GetValueAndLen(&value_len);
          if (value_len > 0)
            my_strlcpy(vendor_id, value, sizeof(vendor_id));
        }
      }
      sys_close(fd);
    }

    // make sure we got everything we wanted
    for (const CpuInfoEntry& entry : cpu_info_table) {
      if (!entry.found) {
        return false;
      }
    }
    // cpu_info_table[0] holds the last cpu id listed in /proc/cpuinfo,
    // assuming this is the highest id, change it to the number of CPUs
    // by adding one.
    cpu_info_table[0].value++;

    sys_info->number_of_processors = cpu_info_table[0].value;
#if defined(__i386__) || defined(__x86_64__)
    sys_info->processor_level      = cpu_info_table[3].value;
    sys_info->processor_revision   = cpu_info_table[1].value << 8 |
                                     cpu_info_table[2].value;
#endif

    if (vendor_id[0] != '\0') {
      my_memcpy(sys_info->cpu.x86_cpu_info.vendor_id, vendor_id,
                sizeof(sys_info->cpu.x86_cpu_info.vendor_id));
    }
    return true;
  }
#elif defined(__arm__) || defined(__aarch64__)
  bool WriteCPUInformation(MDRawSystemInfo* sys_info) {
    // The CPUID value is broken up in several entries in /proc/cpuinfo.
    // This table is used to rebuild it from the entries.
    const struct CpuIdEntry {
      const char* field;
      char        format;
      char        bit_lshift;
      char        bit_length;
    } cpu_id_entries[] = {
      { "CPU implementer", 'x', 24, 8 },
      { "CPU variant", 'x', 20, 4 },
      { "CPU part", 'x', 4, 12 },
      { "CPU revision", 'd', 0, 4 },
    };

    // The ELF hwcaps are listed in the "Features" entry as textual tags.
    // This table is used to rebuild them.
    const struct CpuFeaturesEntry {
      const char* tag;
      uint32_t hwcaps;
    } cpu_features_entries[] = {
#if defined(__arm__)
      { "swp",  MD_CPU_ARM_ELF_HWCAP_SWP },
      { "half", MD_CPU_ARM_ELF_HWCAP_HALF },
      { "thumb", MD_CPU_ARM_ELF_HWCAP_THUMB },
      { "26bit", MD_CPU_ARM_ELF_HWCAP_26BIT },
      { "fastmult", MD_CPU_ARM_ELF_HWCAP_FAST_MULT },
      { "fpa", MD_CPU_ARM_ELF_HWCAP_FPA },
      { "vfp", MD_CPU_ARM_ELF_HWCAP_VFP },
      { "edsp", MD_CPU_ARM_ELF_HWCAP_EDSP },
      { "java", MD_CPU_ARM_ELF_HWCAP_JAVA },
      { "iwmmxt", MD_CPU_ARM_ELF_HWCAP_IWMMXT },
      { "crunch", MD_CPU_ARM_ELF_HWCAP_CRUNCH },
      { "thumbee", MD_CPU_ARM_ELF_HWCAP_THUMBEE },
      { "neon", MD_CPU_ARM_ELF_HWCAP_NEON },
      { "vfpv3", MD_CPU_ARM_ELF_HWCAP_VFPv3 },
      { "vfpv3d16", MD_CPU_ARM_ELF_HWCAP_VFPv3D16 },
      { "tls", MD_CPU_ARM_ELF_HWCAP_TLS },
      { "vfpv4", MD_CPU_ARM_ELF_HWCAP_VFPv4 },
      { "idiva", MD_CPU_ARM_ELF_HWCAP_IDIVA },
      { "idivt", MD_CPU_ARM_ELF_HWCAP_IDIVT },
      { "idiv", MD_CPU_ARM_ELF_HWCAP_IDIVA | MD_CPU_ARM_ELF_HWCAP_IDIVT },
#elif defined(__aarch64__)
      // No hwcaps on aarch64.
#endif
    };

    // processor_architecture should always be set, do this first
    sys_info->processor_architecture =
#if defined(__aarch64__)
        MD_CPU_ARCHITECTURE_ARM64_OLD;
#else
        MD_CPU_ARCHITECTURE_ARM;
#endif

    // /proc/cpuinfo is not readable under various sandboxed environments
    // (e.g. Android services with the android:isolatedProcess attribute)
    // prepare for this by setting default values now, which will be
    // returned when this happens.
    //
    // Note: Bogus values are used to distinguish between failures (to
    //       read /sys and /proc files) and really badly configured kernels.
    sys_info->number_of_processors = 0;
    sys_info->processor_level = 1U;  // There is no ARMv1
    sys_info->processor_revision = 42;
    sys_info->cpu.arm_cpu_info.cpuid = 0;
    sys_info->cpu.arm_cpu_info.elf_hwcaps = 0;

    // Counting the number of CPUs involves parsing two sysfs files,
    // because the content of /proc/cpuinfo will only mirror the number
    // of 'online' cores, and thus will vary with time.
    // See http://www.kernel.org/doc/Documentation/cputopology.txt
    {
      CpuSet cpus_present;
      CpuSet cpus_possible;

      int fd = sys_open("/sys/devices/system/cpu/present", O_RDONLY, 0);
      if (fd >= 0) {
        cpus_present.ParseSysFile(fd);
        sys_close(fd);

        fd = sys_open("/sys/devices/system/cpu/possible", O_RDONLY, 0);
        if (fd >= 0) {
          cpus_possible.ParseSysFile(fd);
          sys_close(fd);

          cpus_present.IntersectWith(cpus_possible);
          int cpu_count = std::min(255, cpus_present.GetCount());
          sys_info->number_of_processors = static_cast<uint8_t>(cpu_count);
        }
      }
    }

    // Parse /proc/cpuinfo to reconstruct the CPUID value, as well
    // as the ELF hwcaps field. For the latter, it would be easier to
    // read /proc/self/auxv but unfortunately, this file is not always
    // readable from regular Android applications on later versions
    // (>= 4.1) of the Android platform.
    const int fd = sys_open("/proc/cpuinfo", O_RDONLY, 0);
    if (fd < 0) {
      // Do not return false here to allow the minidump generation
      // to happen properly.
      return true;
    }

    {
      PageAllocator allocator;
      ProcCpuInfoReader* const reader =
          new(allocator) ProcCpuInfoReader(fd);
      const char* field;
      while (reader->GetNextField(&field)) {
        for (const CpuIdEntry& entry : cpu_id_entries) {
          if (my_strcmp(entry.field, field) != 0)
            continue;
          uintptr_t result = 0;
          const char* value = reader->GetValue();
          const char* p = value;
          if (value[0] == '0' && value[1] == 'x') {
            p = my_read_hex_ptr(&result, value+2);
          } else if (entry.format == 'x') {
            p = my_read_hex_ptr(&result, value);
          } else {
            p = my_read_decimal_ptr(&result, value);
          }
          if (p == value)
            continue;

          result &= (1U << entry.bit_length)-1;
          result <<= entry.bit_lshift;
          sys_info->cpu.arm_cpu_info.cpuid |=
              static_cast<uint32_t>(result);
        }
#if defined(__arm__)
        // Get the architecture version from the "Processor" field.
        // Note that it is also available in the "CPU architecture" field,
        // however, some existing kernels are misconfigured and will report
        // invalid values here (e.g. 6, while the CPU is ARMv7-A based).
        // The "Processor" field doesn't have this issue.
        if (!my_strcmp(field, "Processor")) {
          size_t value_len;
          const char* value = reader->GetValueAndLen(&value_len);
          // Expected format: <text> (v<level><endian>)
          // Where <text> is some text like "ARMv7 Processor rev 2"
          // and <level> is a decimal corresponding to the ARM
          // architecture number. <endian> is either 'l' or 'b'
          // and corresponds to the endianess, it is ignored here.
          while (value_len > 0 && my_isspace(value[value_len-1]))
            value_len--;

          size_t nn = value_len;
          while (nn > 0 && value[nn-1] != '(')
            nn--;
          if (nn > 0 && value[nn] == 'v') {
            uintptr_t arch_level = 5;
            my_read_decimal_ptr(&arch_level, value + nn + 1);
            sys_info->processor_level = static_cast<uint16_t>(arch_level);
          }
        }
#elif defined(__aarch64__)
        // The aarch64 architecture does not provide the architecture level
        // in the Processor field, so we instead check the "CPU architecture"
        // field.
        if (!my_strcmp(field, "CPU architecture")) {
          uintptr_t arch_level = 0;
          const char* value = reader->GetValue();
          const char* p = value;
          p = my_read_decimal_ptr(&arch_level, value);
          if (p == value)
            continue;
          sys_info->processor_level = static_cast<uint16_t>(arch_level);
        }
#endif
        // Rebuild the ELF hwcaps from the 'Features' field.
        if (!my_strcmp(field, "Features")) {
          size_t value_len;
          const char* value = reader->GetValueAndLen(&value_len);

          // Parse each space-separated tag.
          while (value_len > 0) {
            const char* tag = value;
            size_t tag_len = value_len;
            const char* p = my_strchr(tag, ' ');
            if (p) {
              tag_len = static_cast<size_t>(p - tag);
              value += tag_len + 1;
              value_len -= tag_len + 1;
            } else {
              tag_len = strlen(tag);
              value_len = 0;
            }
            for (const CpuFeaturesEntry& entry : cpu_features_entries) {
              if (tag_len == strlen(entry.tag) &&
                  !memcmp(tag, entry.tag, tag_len)) {
                sys_info->cpu.arm_cpu_info.elf_hwcaps |= entry.hwcaps;
                break;
              }
            }
          }
        }
      }
      sys_close(fd);
    }

    return true;
  }
#elif defined(__riscv)
  bool WriteCPUInformation(MDRawSystemInfo* sys_info) {
    // processor_architecture should always be set, do this first
# if __riscv_xlen == 32
    sys_info->processor_architecture = MD_CPU_ARCHITECTURE_RISCV;
# elif __riscv_xlen == 64
    sys_info->processor_architecture = MD_CPU_ARCHITECTURE_RISCV64;
# else
#  error "Unexpected __riscv_xlen"
# endif

    // /proc/cpuinfo is not readable under various sandboxed environments
    // (e.g. Android services with the android:isolatedProcess attribute)
    // prepare for this by setting default values now, which will be
    // returned when this happens.
    //
    // Note: Bogus values are used to distinguish between failures (to
    //       read /sys and /proc files) and really badly configured kernels.
    sys_info->number_of_processors = 0;
    sys_info->processor_level = 0U;
    sys_info->processor_revision = 42;
    sys_info->cpu.other_cpu_info.processor_features[0] = 0;
    sys_info->cpu.other_cpu_info.processor_features[1] = 0;

    // Counting the number of CPUs involves parsing two sysfs files,
    // because the content of /proc/cpuinfo will only mirror the number
    // of 'online' cores, and thus will vary with time.
    // See http://www.kernel.org/doc/Documentation/cputopology.txt
    {
      CpuSet cpus_present;
      CpuSet cpus_possible;

      int fd = sys_open("/sys/devices/system/cpu/present",
                        O_RDONLY | O_CLOEXEC, 0);
      if (fd >= 0) {
        cpus_present.ParseSysFile(fd);
        sys_close(fd);

        fd = sys_open("/sys/devices/system/cpu/possible",
                      O_RDONLY | O_CLOEXEC, 0);
        if (fd >= 0) {
          cpus_possible.ParseSysFile(fd);
          sys_close(fd);

          cpus_present.IntersectWith(cpus_possible);
          int cpu_count = std::min(255, cpus_present.GetCount());
          sys_info->number_of_processors = static_cast<uint8_t>(cpu_count);
        }
      }
    }

    return true;
  }
#else
#  error "Unsupported CPU"
#endif

  bool WriteFile(MDLocationDescriptor* result, const char* filename) {
    const int fd = sys_open(filename, O_RDONLY, 0);
    if (fd < 0)
      return false;

    // We can't stat the files because several of the files that we want to
    // read are kernel seqfiles, which always have a length of zero. So we have
    // to read as much as we can into a buffer.
    static const unsigned kBufSize = 1024 - 2*sizeof(void*);
    struct Buffers {
      Buffers* next;
      size_t len;
      uint8_t data[kBufSize];
    }* buffers = reinterpret_cast<Buffers*>(Alloc(sizeof(Buffers)));
    buffers->next = NULL;
    buffers->len = 0;

    size_t total = 0;
    for (Buffers* bufptr = buffers;;) {
      ssize_t r;
      do {
        r = sys_read(fd, &bufptr->data[bufptr->len], kBufSize - bufptr->len);
      } while (r == -1 && errno == EINTR);

      if (r < 1)
        break;

      total += r;
      bufptr->len += r;
      if (bufptr->len == kBufSize) {
        bufptr->next = reinterpret_cast<Buffers*>(Alloc(sizeof(Buffers)));
        bufptr = bufptr->next;
        bufptr->next = NULL;
        bufptr->len = 0;
      }
    }
    sys_close(fd);

    if (!total)
      return false;

    UntypedMDRVA memory(&minidump_writer_);
    if (!memory.Allocate(total))
      return false;
    for (MDRVA pos = memory.position(); buffers; buffers = buffers->next) {
      // Check for special case of a zero-length buffer.  This should only
      // occur if a file's size happens to be a multiple of the buffer's
      // size, in which case the final sys_read() will have resulted in
      // zero bytes being read after the final buffer was just allocated.
      if (buffers->len == 0) {
        // This can only occur with final buffer.
        assert(buffers->next == NULL);
        continue;
      }
      memory.Copy(pos, &buffers->data, buffers->len);
      pos += buffers->len;
    }
    *result = memory.location();
    return true;
  }

  bool WriteOSInformation(MDRawSystemInfo* sys_info) {
#if defined(__ANDROID__)
    sys_info->platform_id = MD_OS_ANDROID;
#else
    sys_info->platform_id = MD_OS_LINUX;
#endif

    struct utsname uts;
    if (uname(&uts))
      return false;

    static const size_t buf_len = 512;
    char buf[buf_len] = {0};
    size_t space_left = buf_len - 1;
    const char* info_table[] = {
      uts.sysname,
      uts.release,
      uts.version,
      uts.machine,
      NULL
    };
    bool first_item = true;
    for (const char** cur_info = info_table; *cur_info; cur_info++) {
      static const char separator[] = " ";
      size_t separator_len = sizeof(separator) - 1;
      size_t info_len = my_strlen(*cur_info);
      if (info_len == 0)
        continue;

      if (space_left < info_len + (first_item ? 0 : separator_len))
        break;

      if (!first_item) {
        my_strlcat(buf, separator, sizeof(buf));
        space_left -= separator_len;
      }

      first_item = false;
      my_strlcat(buf, *cur_info, sizeof(buf));
      space_left -= info_len;
    }

    MDLocationDescriptor location;
    if (!minidump_writer_.WriteString(buf, 0, &location))
      return false;
    sys_info->csd_version_rva = location.rva;

    return true;
  }

  bool WriteProcFile(MDLocationDescriptor* result, pid_t pid,
                     const char* filename) {
    char buf[NAME_MAX];
    if (!dumper_->BuildProcPath(buf, pid, filename))
      return false;
    return WriteFile(result, buf);
  }

  // Only one of the 2 member variables below should be set to a valid value.
  const int fd_;  // File descriptor where the minidum should be written.
  const char* path_;  // Path to the file where the minidum should be written.

  const ucontext_t* const ucontext_;  // also from the signal handler
#if GOOGLE_BREAKPAD_CRASH_CONTEXT_HAS_FLOAT_STATE
  const google_breakpad::fpstate_t* const float_state_;  // ditto
#endif
  LinuxDumper* dumper_;
  MinidumpFileWriter minidump_writer_;
  off_t minidump_size_limit_;
  MDLocationDescriptor crashing_thread_context_;
  // Blocks of memory written to the dump. These are all currently
  // written while writing the thread list stream, but saved here
  // so a memory list stream can be written afterwards.
  wasteful_vector<MDMemoryDescriptor> memory_blocks_;
  // Additional information about some mappings provided by the caller.
  const MappingList& mapping_list_;
  // Additional memory regions to be included in the dump,
  // provided by the caller.
  const AppMemoryList& app_memory_list_;
  // If set, skip recording any threads that do not reference the
  // mapping containing principal_mapping_address_.
  bool skip_stacks_if_mapping_unreferenced_;
  uintptr_t principal_mapping_address_;
  const MappingInfo* principal_mapping_;
  // If true, apply stack sanitization to stored stack data.
  bool sanitize_stacks_;
};


bool WriteMinidumpImpl(const char* minidump_path,
                       int minidump_fd,
                       off_t minidump_size_limit,
                       pid_t crashing_process,
                       const void* blob, size_t blob_size,
                       const MappingList& mappings,
                       const AppMemoryList& appmem,
                       bool skip_stacks_if_mapping_unreferenced,
                       uintptr_t principal_mapping_address,
                       bool sanitize_stacks) {
  LinuxPtraceDumper dumper(crashing_process);
  const ExceptionHandler::CrashContext* context = NULL;
  if (blob) {
    if (blob_size != sizeof(ExceptionHandler::CrashContext))
      return false;
    context = reinterpret_cast<const ExceptionHandler::CrashContext*>(blob);
    dumper.SetCrashInfoFromSigInfo(context->siginfo);
    dumper.set_crash_thread(context->tid);
  }
  MinidumpWriter writer(minidump_path, minidump_fd, context, mappings,
                        appmem, skip_stacks_if_mapping_unreferenced,
                        principal_mapping_address, sanitize_stacks, &dumper);
  // Set desired limit for file size of minidump (-1 means no limit).
  writer.set_minidump_size_limit(minidump_size_limit);
  if (!writer.Init())
    return false;
  return writer.Dump();
}

}  // namespace

namespace google_breakpad {

bool WriteMinidump(const char* minidump_path, pid_t crashing_process,
                   const void* blob, size_t blob_size,
                   bool skip_stacks_if_mapping_unreferenced,
                   uintptr_t principal_mapping_address,
                   bool sanitize_stacks) {
  return WriteMinidumpImpl(minidump_path, -1, -1,
                           crashing_process, blob, blob_size,
                           MappingList(), AppMemoryList(),
                           skip_stacks_if_mapping_unreferenced,
                           principal_mapping_address,
                           sanitize_stacks);
}

bool WriteMinidump(int minidump_fd, pid_t crashing_process,
                   const void* blob, size_t blob_size,
                   bool skip_stacks_if_mapping_unreferenced,
                   uintptr_t principal_mapping_address,
                   bool sanitize_stacks) {
  return WriteMinidumpImpl(NULL, minidump_fd, -1,
                           crashing_process, blob, blob_size,
                           MappingList(), AppMemoryList(),
                           skip_stacks_if_mapping_unreferenced,
                           principal_mapping_address,
                           sanitize_stacks);
}

bool WriteMinidump(const char* minidump_path, pid_t process,
                   pid_t process_blamed_thread) {
  LinuxPtraceDumper dumper(process);
  // MinidumpWriter will set crash address
  dumper.set_crash_signal(MD_EXCEPTION_CODE_LIN_DUMP_REQUESTED);
  dumper.set_crash_thread(process_blamed_thread);
  MappingList mapping_list;
  AppMemoryList app_memory_list;
  MinidumpWriter writer(minidump_path, -1, NULL, mapping_list,
                        app_memory_list, false, 0, false, &dumper);
  if (!writer.Init())
    return false;
  return writer.Dump();
}

bool WriteMinidump(const char* minidump_path, pid_t crashing_process,
                   const void* blob, size_t blob_size,
                   const MappingList& mappings,
                   const AppMemoryList& appmem,
                   bool skip_stacks_if_mapping_unreferenced,
                   uintptr_t principal_mapping_address,
                   bool sanitize_stacks) {
  return WriteMinidumpImpl(minidump_path, -1, -1, crashing_process,
                           blob, blob_size,
                           mappings, appmem,
                           skip_stacks_if_mapping_unreferenced,
                           principal_mapping_address,
                           sanitize_stacks);
}

bool WriteMinidump(int minidump_fd, pid_t crashing_process,
                   const void* blob, size_t blob_size,
                   const MappingList& mappings,
                   const AppMemoryList& appmem,
                   bool skip_stacks_if_mapping_unreferenced,
                   uintptr_t principal_mapping_address,
                   bool sanitize_stacks) {
  return WriteMinidumpImpl(NULL, minidump_fd, -1, crashing_process,
                           blob, blob_size,
                           mappings, appmem,
                           skip_stacks_if_mapping_unreferenced,
                           principal_mapping_address,
                           sanitize_stacks);
}

bool WriteMinidump(const char* minidump_path, off_t minidump_size_limit,
                   pid_t crashing_process,
                   const void* blob, size_t blob_size,
                   const MappingList& mappings,
                   const AppMemoryList& appmem,
                   bool skip_stacks_if_mapping_unreferenced,
                   uintptr_t principal_mapping_address,
                   bool sanitize_stacks) {
  return WriteMinidumpImpl(minidump_path, -1, minidump_size_limit,
                           crashing_process, blob, blob_size,
                           mappings, appmem,
                           skip_stacks_if_mapping_unreferenced,
                           principal_mapping_address,
                           sanitize_stacks);
}

bool WriteMinidump(int minidump_fd, off_t minidump_size_limit,
                   pid_t crashing_process,
                   const void* blob, size_t blob_size,
                   const MappingList& mappings,
                   const AppMemoryList& appmem,
                   bool skip_stacks_if_mapping_unreferenced,
                   uintptr_t principal_mapping_address,
                   bool sanitize_stacks) {
  return WriteMinidumpImpl(NULL, minidump_fd, minidump_size_limit,
                           crashing_process, blob, blob_size,
                           mappings, appmem,
                           skip_stacks_if_mapping_unreferenced,
                           principal_mapping_address,
                           sanitize_stacks);
}

bool WriteMinidump(const char* filename,
                   const MappingList& mappings,
                   const AppMemoryList& appmem,
                   LinuxDumper* dumper) {
  MinidumpWriter writer(filename, -1, NULL, mappings, appmem,
                        false, 0, false, dumper);
  if (!writer.Init())
    return false;
  return writer.Dump();
}

}  // namespace google_breakpad
