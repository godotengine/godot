// Copyright (c) 2010, Google Inc.
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

// linux_dumper.h: Define the google_breakpad::LinuxDumper class, which
// is a base class for extracting information of a crashed process. It
// was originally a complete implementation using the ptrace API, but
// has been refactored to allow derived implementations supporting both
// ptrace and core dump. A portion of the original implementation is now
// in google_breakpad::LinuxPtraceDumper (see linux_ptrace_dumper.h for
// details).

#ifndef CLIENT_LINUX_MINIDUMP_WRITER_LINUX_DUMPER_H_
#define CLIENT_LINUX_MINIDUMP_WRITER_LINUX_DUMPER_H_

#include <assert.h>
#include <elf.h>
#if defined(__ANDROID__)
#include <link.h>
#endif
#include <linux/limits.h>
#include <stdint.h>
#include <sys/types.h>
#include <sys/user.h>

#include <vector>

#include "client/linux/dump_writer_common/mapping_info.h"
#include "client/linux/dump_writer_common/thread_info.h"
#include "common/linux/file_id.h"
#include "common/memory_allocator.h"
#include "google_breakpad/common/minidump_format.h"

namespace google_breakpad {

// Typedef for our parsing of the auxv variables in /proc/pid/auxv.
#if defined(__i386) || defined(__ARM_EABI__) || \
 (defined(__mips__) && _MIPS_SIM == _ABIO32)
typedef Elf32_auxv_t elf_aux_entry;
#elif defined(__x86_64) || defined(__aarch64__) || \
     (defined(__mips__) && _MIPS_SIM != _ABIO32)
typedef Elf64_auxv_t elf_aux_entry;
#endif

typedef __typeof__(((elf_aux_entry*) 0)->a_un.a_val) elf_aux_val_t;

// When we find the VDSO mapping in the process's address space, this
// is the name we use for it when writing it to the minidump.
// This should always be less than NAME_MAX!
const char kLinuxGateLibraryName[] = "linux-gate.so";

class LinuxDumper {
 public:
  // The |root_prefix| is prepended to mapping paths before opening them, which
  // is useful if the crash originates from a chroot.
  explicit LinuxDumper(pid_t pid, const char* root_prefix = "");

  virtual ~LinuxDumper();

  // Parse the data for |threads| and |mappings|.
  virtual bool Init();

  // Take any actions that could not be taken in Init(). LateInit() is
  // called after all other caller's initialization is complete, and in
  // particular after it has called ThreadsSuspend(), so that ptrace is
  // available.
  virtual bool LateInit();

  // Return true if the dumper performs a post-mortem dump.
  virtual bool IsPostMortem() const = 0;

  // Suspend/resume all threads in the given process.
  virtual bool ThreadsSuspend() = 0;
  virtual bool ThreadsResume() = 0;

  // Read information about the |index|-th thread of |threads_|.
  // Returns true on success. One must have called |ThreadsSuspend| first.
  virtual bool GetThreadInfoByIndex(size_t index, ThreadInfo* info) = 0;

  size_t GetMainThreadIndex() const {
    for (size_t i = 0; i < threads_.size(); ++i) {
      if (threads_[i] == pid_) return i;
    }
    return -1u;
  }

  // These are only valid after a call to |Init|.
  const wasteful_vector<pid_t>& threads() { return threads_; }
  const wasteful_vector<MappingInfo*>& mappings() { return mappings_; }
  const MappingInfo* FindMapping(const void* address) const;
  // Find the mapping which the given memory address falls in. Unlike
  // FindMapping, this method uses the unadjusted mapping address
  // ranges from the kernel, rather than the ranges that have had the
  // load bias applied.
  const MappingInfo* FindMappingNoBias(uintptr_t address) const;
  const wasteful_vector<elf_aux_val_t>& auxv() { return auxv_; }

  // Find a block of memory to take as the stack given the top of stack pointer.
  //   stack: (output) the lowest address in the memory area
  //   stack_len: (output) the length of the memory area
  //   stack_top: the current top of the stack
  bool GetStackInfo(const void** stack, size_t* stack_len, uintptr_t stack_top);

  // Sanitize a copy of the stack by overwriting words that are not
  // pointers with a sentinel (0x0defaced).
  //   stack_copy: a copy of the stack to sanitize. |stack_copy| might
  //               not be word aligned, but it represents word aligned
  //               data copied from another location.
  //   stack_len: the length of the allocation pointed to by |stack_copy|.
  //   stack_pointer: the address of the stack pointer (used to locate
  //                  the stack mapping, as an optimization).
  //   sp_offset: the offset relative to stack_copy that reflects the
  //              current value of the stack pointer.
  void SanitizeStackCopy(uint8_t* stack_copy, size_t stack_len,
                         uintptr_t stack_pointer, uintptr_t sp_offset);

  // Test whether |stack_copy| contains a pointer-aligned word that
  // could be an address within a given mapping.
  //   stack_copy: a copy of the stack to check. |stack_copy| might
  //               not be word aligned, but it represents word aligned
  //               data copied from another location.
  //   stack_len: the length of the allocation pointed to by |stack_copy|.
  //   sp_offset: the offset relative to stack_copy that reflects the
  //              current value of the stack pointer.
  //   mapping: the mapping against which to test stack words.
  bool StackHasPointerToMapping(const uint8_t* stack_copy, size_t stack_len,
                                uintptr_t sp_offset,
                                const MappingInfo& mapping);

  PageAllocator* allocator() { return &allocator_; }

  // Copy content of |length| bytes from a given process |child|,
  // starting from |src|, into |dest|. Returns true on success.
  virtual bool CopyFromProcess(void* dest, pid_t child, const void* src,
                               size_t length) = 0;

  // Builds a proc path for a certain pid for a node (/proc/<pid>/<node>).
  // |path| is a character array of at least NAME_MAX bytes to return the
  // result.|node| is the final node without any slashes. Returns true on
  // success.
  virtual bool BuildProcPath(char* path, pid_t pid, const char* node) const = 0;

  // Generate a File ID from the .text section of a mapped entry.
  // If not a member, mapping_id is ignored. This method can also manipulate the
  // |mapping|.name to truncate "(deleted)" from the file name if necessary.
  bool ElfFileIdentifierForMapping(const MappingInfo& mapping,
                                   bool member,
                                   unsigned int mapping_id,
                                   wasteful_vector<uint8_t>& identifier);

  void SetCrashInfoFromSigInfo(const siginfo_t& siginfo);

  uintptr_t crash_address() const { return crash_address_; }
  void set_crash_address(uintptr_t crash_address) {
    crash_address_ = crash_address;
  }

  int crash_signal() const { return crash_signal_; }
  void set_crash_signal(int crash_signal) { crash_signal_ = crash_signal; }
  const char* GetCrashSignalString() const;

  void set_crash_signal_code(int code) { crash_signal_code_ = code; }
  int crash_signal_code() const { return crash_signal_code_; }

  void set_crash_exception_info(const std::vector<uint64_t>& exception_info) {
    assert(exception_info.size() <= MD_EXCEPTION_MAXIMUM_PARAMETERS);
    crash_exception_info_ = exception_info;
  }
  const std::vector<uint64_t>& crash_exception_info() const {
    return crash_exception_info_;
  }

  pid_t crash_thread() const { return crash_thread_; }
  void set_crash_thread(pid_t crash_thread) { crash_thread_ = crash_thread; }

  // Concatenates the |root_prefix_| and |mapping| path. Writes into |path| and
  // returns true unless the string is too long.
  bool GetMappingAbsolutePath(const MappingInfo& mapping,
                              char path[PATH_MAX]) const;

  // Extracts the effective path and file name of from |mapping|. In most cases
  // the effective name/path are just the mapping's path and basename. In some
  // other cases, however, a library can be mapped from an archive (e.g., when
  // loading .so libs from an apk on Android) and this method is able to
  // reconstruct the original file name.
  void GetMappingEffectiveNameAndPath(const MappingInfo& mapping,
                                      char* file_path,
                                      size_t file_path_size,
                                      char* file_name,
                                      size_t file_name_size);

 protected:
  bool ReadAuxv();

  virtual bool EnumerateMappings();

  virtual bool EnumerateThreads() = 0;

  // For the case where a running program has been deleted, it'll show up in
  // /proc/pid/maps as "/path/to/program (deleted)". If this is the case, then
  // see if '/path/to/program (deleted)' matches /proc/pid/exe and return
  // /proc/pid/exe in |path| so ELF identifier generation works correctly. This
  // also checks to see if '/path/to/program (deleted)' exists, so it does not
  // get fooled by a poorly named binary.
  // For programs that don't end with ' (deleted)', this is a no-op.
  // This assumes |path| is a buffer with length NAME_MAX.
  // Returns true if |path| is modified.
  bool HandleDeletedFileInMapping(char* path) const;

   // ID of the crashed process.
  const pid_t pid_;

  // Path of the root directory to which mapping paths are relative.
  const char* const root_prefix_;

  // Virtual address at which the process crashed.
  uintptr_t crash_address_;

  // Signal that terminated the crashed process.
  int crash_signal_;

  // The code associated with |crash_signal_|.
  int crash_signal_code_;

  // The additional fields associated with |crash_signal_|.
  std::vector<uint64_t> crash_exception_info_;

  // ID of the crashed thread.
  pid_t crash_thread_;

  mutable PageAllocator allocator_;

  // IDs of all the threads.
  wasteful_vector<pid_t> threads_;

  // Info from /proc/<pid>/maps.
  wasteful_vector<MappingInfo*> mappings_;

  // Info from /proc/<pid>/auxv
  wasteful_vector<elf_aux_val_t> auxv_;

#if defined(__ANDROID__)
 private:
  // Android M and later support packed ELF relocations in shared libraries.
  // Packing relocations changes the vaddr of the LOAD segments, such that
  // the effective load bias is no longer the same as the start address of
  // the memory mapping containing the executable parts of the library. The
  // packing is applied to the stripped library run on the target, but not to
  // any other library, and in particular not to the library used to generate
  // breakpad symbols. As a result, we need to adjust the |start_addr| for
  // any mapping that results from a shared library that contains Android
  // packed relocations, so that it properly represents the effective library
  // load bias. The following functions support this adjustment.

  // Check that a given mapping at |start_addr| is for an ELF shared library.
  // If it is, place the ELF header in |ehdr| and return true.
  // The first LOAD segment in an ELF shared library has offset zero, so the
  // ELF file header is at the start of this map entry, and in already mapped
  // memory.
  bool GetLoadedElfHeader(uintptr_t start_addr, ElfW(Ehdr)* ehdr);

  // For the ELF file mapped at |start_addr|, iterate ELF program headers to
  // find the min vaddr of all program header LOAD segments, the vaddr for
  // the DYNAMIC segment, and a count of DYNAMIC entries. Return values in
  // |min_vaddr_ptr|, |dyn_vaddr_ptr|, and |dyn_count_ptr|.
  // The program header table is also in already mapped memory.
  void ParseLoadedElfProgramHeaders(ElfW(Ehdr)* ehdr,
                                    uintptr_t start_addr,
                                    uintptr_t* min_vaddr_ptr,
                                    uintptr_t* dyn_vaddr_ptr,
                                    size_t* dyn_count_ptr);

  // Search the DYNAMIC tags for the ELF file with the given |load_bias|, and
  // return true if the tags indicate that the file contains Android packed
  // relocations. Dynamic tags are found at |dyn_vaddr| past the |load_bias|.
  bool HasAndroidPackedRelocations(uintptr_t load_bias,
                                   uintptr_t dyn_vaddr,
                                   size_t dyn_count);

  // If the ELF file mapped at |start_addr| contained Android packed
  // relocations, return the load bias that the system linker (or Chromium
  // crazy linker) will have used. If the file did not contain Android
  // packed relocations, returns |start_addr|, indicating that no adjustment
  // is necessary.
  // The effective load bias is |start_addr| adjusted downwards by the
  // min vaddr in the library LOAD segments.
  uintptr_t GetEffectiveLoadBias(ElfW(Ehdr)* ehdr, uintptr_t start_addr);

  // Called from LateInit(). Iterates |mappings_| and rewrites the |start_addr|
  // field of any that represent ELF shared libraries with Android packed
  // relocations, so that |start_addr| is the load bias that the system linker
  // (or Chromium crazy linker) used. This value matches the addresses produced
  // when the non-relocation-packed library is used for breakpad symbol
  // generation.
  void LatePostprocessMappings();
#endif  // __ANDROID__
};

}  // namespace google_breakpad

#endif  // CLIENT_LINUX_HANDLER_LINUX_DUMPER_H_
