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

// linux_dumper.cc: Implement google_breakpad::LinuxDumper.
// See linux_dumper.h for details.

// This code deals with the mechanics of getting information about a crashed
// process. Since this code may run in a compromised address space, the same
// rules apply as detailed at the top of minidump_writer.h: no libc calls and
// use the alternative allocator.

#ifdef HAVE_CONFIG_H
#include <config.h>  // Must come first
#endif

#include "client/linux/minidump_writer/linux_dumper.h"

#include <assert.h>
#include <elf.h>
#include <fcntl.h>
#include <limits.h>
#include <stddef.h>
#include <string.h>

#include "client/linux/minidump_writer/line_reader.h"
#include "common/linux/elfutils.h"
#include "common/linux/file_id.h"
#include "common/linux/linux_libc_support.h"
#include "common/linux/memory_mapped_file.h"
#include "common/linux/safe_readlink.h"
#include "google_breakpad/common/minidump_exception_linux.h"
#include "third_party/lss/linux_syscall_support.h"

using google_breakpad::elf::FileID;

#if defined(__ANDROID__)

// Android packed relocations definitions are not yet available from the
// NDK header files, so we have to provide them manually here.
#ifndef DT_LOOS
#define DT_LOOS 0x6000000d
#endif
#ifndef DT_ANDROID_REL
static const int DT_ANDROID_REL = DT_LOOS + 2;
#endif
#ifndef DT_ANDROID_RELA
static const int DT_ANDROID_RELA = DT_LOOS + 4;
#endif

#endif  // __ANDROID __

static const char kMappedFileUnsafePrefix[] = "/dev/";
static const char kDeletedSuffix[] = " (deleted)";

inline static bool IsMappedFileOpenUnsafe(
    const google_breakpad::MappingInfo& mapping) {
  // It is unsafe to attempt to open a mapped file that lives under /dev,
  // because the semantics of the open may be driver-specific so we'd risk
  // hanging the crash dumper. And a file in /dev/ almost certainly has no
  // ELF file identifier anyways.
  return my_strncmp(mapping.name,
                    kMappedFileUnsafePrefix,
                    sizeof(kMappedFileUnsafePrefix) - 1) == 0;
}

namespace google_breakpad {

namespace {

bool MappingContainsAddress(const MappingInfo& mapping, uintptr_t address) {
  return mapping.system_mapping_info.start_addr <= address &&
         address < mapping.system_mapping_info.end_addr;
}

#if defined(__CHROMEOS__)

// Recover memory mappings before writing dump on ChromeOS
//
// On Linux, breakpad relies on /proc/[pid]/maps to associate symbols from
// addresses. ChromeOS' hugepage implementation replaces some segments with
// anonymous private pages, which is a restriction of current implementation
// in Linux kernel at the time of writing. Thus, breakpad can no longer
// symbolize addresses from those text segments replaced with hugepages.
//
// This postprocess tries to recover the mappings. Because hugepages are always
// inserted in between some .text sections, it tries to infer the names and
// offsets of the segments, by looking at segments immediately precede and
// succeed them.
//
// For example, a text segment before hugepage optimization
//   02001000-03002000 r-xp /opt/google/chrome/chrome
//
// can be broken into
//   02001000-02200000 r-xp /opt/google/chrome/chrome
//   02200000-03000000 r-xp
//   03000000-03002000 r-xp /opt/google/chrome/chrome
//
// For more details, see:
// crbug.com/628040 ChromeOS' use of hugepages confuses crash symbolization

// Copied from CrOS' hugepage implementation, which is unlikely to change.
// The hugepage size is 2M.
const unsigned int kHpageShift = 21;
const size_t kHpageSize = (1 << kHpageShift);
const size_t kHpageMask = (~(kHpageSize - 1));

// Find and merge anonymous r-xp segments with surrounding named segments.
// There are two cases:

// Case 1: curr, next
//   curr is anonymous
//   curr is r-xp
//   curr.size >= 2M
//   curr.size is a multiple of 2M.
//   next is backed by some file.
//   curr and next are contiguous.
//   offset(next) == sizeof(curr)
void TryRecoverMappings(MappingInfo* curr, MappingInfo* next) {
  // Merged segments are marked with size = 0.
  if (curr->size == 0 || next->size == 0)
    return;

  if (curr->size >= kHpageSize &&
      curr->exec &&
      (curr->size & kHpageMask) == curr->size &&
      (curr->start_addr & kHpageMask) == curr->start_addr &&
      curr->name[0] == '\0' &&
      next->name[0] != '\0' &&
      curr->start_addr + curr->size == next->start_addr &&
      curr->size == next->offset) {

    // matched
    my_strlcpy(curr->name, next->name, NAME_MAX);
    if (next->exec) {
      // (curr, next)
      curr->size += next->size;
      next->size = 0;
    }
  }
}

// Case 2: prev, curr, next
//   curr is anonymous
//   curr is r-xp
//   curr.size >= 2M
//   curr.size is a multiple of 2M.
//   next and prev are backed by the same file.
//   prev, curr and next are contiguous.
//   offset(next) == offset(prev) + sizeof(prev) + sizeof(curr)
void TryRecoverMappings(MappingInfo* prev, MappingInfo* curr,
                        MappingInfo* next) {
  // Merged segments are marked with size = 0.
  if (prev->size == 0 || curr->size == 0 || next->size == 0)
    return;

  if (curr->size >= kHpageSize &&
      curr->exec &&
      (curr->size & kHpageMask) == curr->size &&
      (curr->start_addr & kHpageMask) == curr->start_addr &&
      curr->name[0] == '\0' &&
      next->name[0] != '\0' &&
      curr->start_addr + curr->size == next->start_addr &&
      prev->start_addr + prev->size == curr->start_addr &&
      my_strncmp(prev->name, next->name, NAME_MAX) == 0 &&
      next->offset == prev->offset + prev->size + curr->size) {

    // matched
    my_strlcpy(curr->name, prev->name, NAME_MAX);
    if (prev->exec) {
      curr->offset = prev->offset;
      curr->start_addr = prev->start_addr;
      if (next->exec) {
        // (prev, curr, next)
        curr->size += prev->size + next->size;
        prev->size = 0;
        next->size = 0;
      } else {
        // (prev, curr), next
        curr->size += prev->size;
        prev->size = 0;
      }
    } else {
      curr->offset = prev->offset + prev->size;
      if (next->exec) {
        // prev, (curr, next)
        curr->size += next->size;
        next->size = 0;
      } else {
        // prev, curr, next
      }
    }
  }
}

// mappings_ is sorted excepted for the first entry.
// This function tries to merge segemnts into the first entry,
// then check for other sorted entries.
// See LinuxDumper::EnumerateMappings().
void CrOSPostProcessMappings(wasteful_vector<MappingInfo*>& mappings) {
  // Find the candidate "next" to first segment, which is the only one that
  // could be out-of-order.
  size_t l = 1;
  size_t r = mappings.size();
  size_t next = mappings.size();
  while (l < r) {
    int m = (l + r) / 2;
    if (mappings[m]->start_addr > mappings[0]->start_addr)
      r = next = m;
    else
      l = m + 1;
  }

  // Shows the range that contains the entry point is
  // [first_start_addr, first_end_addr)
  size_t first_start_addr = mappings[0]->start_addr;
  size_t first_end_addr = mappings[0]->start_addr + mappings[0]->size;

  // Put the out-of-order segment in order.
  std::rotate(mappings.begin(), mappings.begin() + 1, mappings.begin() + next);

  // Iterate through normal, sorted cases.
  // Normal case 1.
  for (size_t i = 0; i < mappings.size() - 1; i++)
    TryRecoverMappings(mappings[i], mappings[i + 1]);

  // Normal case 2.
  for (size_t i = 0; i < mappings.size() - 2; i++)
    TryRecoverMappings(mappings[i], mappings[i + 1], mappings[i + 2]);

  // Collect merged (size == 0) segments.
  size_t f, e;
  for (f = e = 0; e < mappings.size(); e++)
    if (mappings[e]->size > 0)
      mappings[f++] = mappings[e];
  mappings.resize(f);

  // The entry point is in the first mapping. We want to find the location
  // of the entry point after merging segment. To do this, we want to find
  // the mapping that covers the first mapping from the original mapping list.
  // If the mapping is not in the beginning, we move it to the begining via
  // a right rotate by using reverse iterators.
  for (l = 0; l < mappings.size(); l++) {
    if (mappings[l]->start_addr <= first_start_addr
        && (mappings[l]->start_addr + mappings[l]->size >= first_end_addr))
      break;
  }
  if (l > 0) {
    r = mappings.size();
    std::rotate(mappings.rbegin() + r - l - 1, mappings.rbegin() + r - l,
                mappings.rend());
  }
}

#endif  // __CHROMEOS__

}  // namespace

// All interesting auvx entry types are below AT_SYSINFO_EHDR
#define AT_MAX AT_SYSINFO_EHDR

LinuxDumper::LinuxDumper(pid_t pid, const char* root_prefix)
    : pid_(pid),
      root_prefix_(root_prefix),
      crash_address_(0),
      crash_signal_(0),
      crash_signal_code_(0),
      crash_thread_(pid),
      threads_(&allocator_, 8),
      mappings_(&allocator_),
      auxv_(&allocator_, AT_MAX + 1) {
  assert(root_prefix_ && my_strlen(root_prefix_) < PATH_MAX);
  // The passed-in size to the constructor (above) is only a hint.
  // Must call .resize() to do actual initialization of the elements.
  auxv_.resize(AT_MAX + 1);
}

LinuxDumper::~LinuxDumper() {
}

bool LinuxDumper::Init() {
  return ReadAuxv() && EnumerateThreads() && EnumerateMappings();
}

bool LinuxDumper::LateInit() {
#if defined(__ANDROID__)
  LatePostprocessMappings();
#endif

#if defined(__CHROMEOS__)
  CrOSPostProcessMappings(mappings_);
#endif

  return true;
}

bool
LinuxDumper::ElfFileIdentifierForMapping(const MappingInfo& mapping,
                                         bool member,
                                         unsigned int mapping_id,
                                         wasteful_vector<uint8_t>& identifier) {
  assert(!member || mapping_id < mappings_.size());
  if (IsMappedFileOpenUnsafe(mapping))
    return false;

  // Special-case linux-gate because it's not a real file.
  if (my_strcmp(mapping.name, kLinuxGateLibraryName) == 0) {
    void* linux_gate = NULL;
    if (pid_ == sys_getpid()) {
      linux_gate = reinterpret_cast<void*>(mapping.start_addr);
    } else {
      linux_gate = allocator_.Alloc(mapping.size);
      CopyFromProcess(linux_gate, pid_,
                      reinterpret_cast<const void*>(mapping.start_addr),
                      mapping.size);
    }
    return FileID::ElfFileIdentifierFromMappedFile(linux_gate, identifier);
  }

  char filename[PATH_MAX];
  if (!GetMappingAbsolutePath(mapping, filename))
    return false;
  bool filename_modified = HandleDeletedFileInMapping(filename);

  MemoryMappedFile mapped_file(filename, 0);
  if (!mapped_file.data() || mapped_file.size() < SELFMAG)
    return false;

  bool success =
      FileID::ElfFileIdentifierFromMappedFile(mapped_file.data(), identifier);
  if (success && member && filename_modified) {
    mappings_[mapping_id]->name[my_strlen(mapping.name) -
                                sizeof(kDeletedSuffix) + 1] = '\0';
  }

  return success;
}

void LinuxDumper::SetCrashInfoFromSigInfo(const siginfo_t& siginfo) {
  set_crash_address(reinterpret_cast<uintptr_t>(siginfo.si_addr));
  set_crash_signal(siginfo.si_signo);
  set_crash_signal_code(siginfo.si_code);
}

const char* LinuxDumper::GetCrashSignalString() const {
  switch (static_cast<unsigned int>(crash_signal_)) {
    case MD_EXCEPTION_CODE_LIN_SIGHUP:
      return "SIGHUP";
    case MD_EXCEPTION_CODE_LIN_SIGINT:
      return "SIGINT";
    case MD_EXCEPTION_CODE_LIN_SIGQUIT:
      return "SIGQUIT";
    case MD_EXCEPTION_CODE_LIN_SIGILL:
      return "SIGILL";
    case MD_EXCEPTION_CODE_LIN_SIGTRAP:
      return "SIGTRAP";
    case MD_EXCEPTION_CODE_LIN_SIGABRT:
      return "SIGABRT";
    case MD_EXCEPTION_CODE_LIN_SIGBUS:
      return "SIGBUS";
    case MD_EXCEPTION_CODE_LIN_SIGFPE:
      return "SIGFPE";
    case MD_EXCEPTION_CODE_LIN_SIGKILL:
      return "SIGKILL";
    case MD_EXCEPTION_CODE_LIN_SIGUSR1:
      return "SIGUSR1";
    case MD_EXCEPTION_CODE_LIN_SIGSEGV:
      return "SIGSEGV";
    case MD_EXCEPTION_CODE_LIN_SIGUSR2:
      return "SIGUSR2";
    case MD_EXCEPTION_CODE_LIN_SIGPIPE:
      return "SIGPIPE";
    case MD_EXCEPTION_CODE_LIN_SIGALRM:
      return "SIGALRM";
    case MD_EXCEPTION_CODE_LIN_SIGTERM:
      return "SIGTERM";
    case MD_EXCEPTION_CODE_LIN_SIGSTKFLT:
      return "SIGSTKFLT";
    case MD_EXCEPTION_CODE_LIN_SIGCHLD:
      return "SIGCHLD";
    case MD_EXCEPTION_CODE_LIN_SIGCONT:
      return "SIGCONT";
    case MD_EXCEPTION_CODE_LIN_SIGSTOP:
      return "SIGSTOP";
    case MD_EXCEPTION_CODE_LIN_SIGTSTP:
      return "SIGTSTP";
    case MD_EXCEPTION_CODE_LIN_SIGTTIN:
      return "SIGTTIN";
    case MD_EXCEPTION_CODE_LIN_SIGTTOU:
      return "SIGTTOU";
    case MD_EXCEPTION_CODE_LIN_SIGURG:
      return "SIGURG";
    case MD_EXCEPTION_CODE_LIN_SIGXCPU:
      return "SIGXCPU";
    case MD_EXCEPTION_CODE_LIN_SIGXFSZ:
      return "SIGXFSZ";
    case MD_EXCEPTION_CODE_LIN_SIGVTALRM:
      return "SIGVTALRM";
    case MD_EXCEPTION_CODE_LIN_SIGPROF:
      return "SIGPROF";
    case MD_EXCEPTION_CODE_LIN_SIGWINCH:
      return "SIGWINCH";
    case MD_EXCEPTION_CODE_LIN_SIGIO:
      return "SIGIO";
    case MD_EXCEPTION_CODE_LIN_SIGPWR:
      return "SIGPWR";
    case MD_EXCEPTION_CODE_LIN_SIGSYS:
      return "SIGSYS";
    case MD_EXCEPTION_CODE_LIN_DUMP_REQUESTED:
      return "DUMP_REQUESTED";
    default:
      return "UNKNOWN";
  }
}

bool LinuxDumper::GetMappingAbsolutePath(const MappingInfo& mapping,
                                         char path[PATH_MAX]) const {
  return my_strlcpy(path, root_prefix_, PATH_MAX) < PATH_MAX &&
         my_strlcat(path, mapping.name, PATH_MAX) < PATH_MAX;
}

namespace {
// Find the shared object name (SONAME) by examining the ELF information
// for |mapping|. If the SONAME is found copy it into the passed buffer
// |soname| and return true. The size of the buffer is |soname_size|.
// The SONAME will be truncated if it is too long to fit in the buffer.
bool ElfFileSoName(const LinuxDumper& dumper,
    const MappingInfo& mapping, char* soname, size_t soname_size) {
  if (IsMappedFileOpenUnsafe(mapping)) {
    // Not safe
    return false;
  }

  char filename[PATH_MAX];
  if (!dumper.GetMappingAbsolutePath(mapping, filename))
    return false;

  MemoryMappedFile mapped_file(filename, 0);
  if (!mapped_file.data() || mapped_file.size() < SELFMAG) {
    // mmap failed
    return false;
  }

  return ElfFileSoNameFromMappedFile(mapped_file.data(), soname, soname_size);
}

}  // namespace


void LinuxDumper::GetMappingEffectiveNameAndPath(const MappingInfo& mapping,
                                                 char* file_path,
                                                 size_t file_path_size,
                                                 char* file_name,
                                                 size_t file_name_size) {
  my_strlcpy(file_path, mapping.name, file_path_size);

  // Tools such as minidump_stackwalk use the name of the module to look up
  // symbols produced by dump_syms. dump_syms will prefer to use a module's
  // DT_SONAME as the module name, if one exists, and will fall back to the
  // filesystem name of the module.

  // Just use the filesystem name if no SONAME is present.
  if (!ElfFileSoName(*this, mapping, file_name, file_name_size)) {
    //   file_path := /path/to/libname.so
    //   file_name := libname.so
    const char* basename = my_strrchr(file_path, '/');
    basename = basename == NULL ? file_path : (basename + 1);
    my_strlcpy(file_name, basename, file_name_size);
    return;
  }

  if (mapping.exec && mapping.offset != 0) {
    // If an executable is mapped from a non-zero offset, this is likely because
    // the executable was loaded directly from inside an archive file (e.g., an
    // apk on Android).
    // In this case, we append the file_name to the mapped archive path:
    //   file_name := libname.so
    //   file_path := /path/to/ARCHIVE.APK/libname.so
    if (my_strlen(file_path) + 1 + my_strlen(file_name) < file_path_size) {
      my_strlcat(file_path, "/", file_path_size);
      my_strlcat(file_path, file_name, file_path_size);
    }
  } else {
    // Otherwise, replace the basename with the SONAME.
    char* basename = const_cast<char*>(my_strrchr(file_path, '/'));
    if (basename) {
      my_strlcpy(basename + 1, file_name,
                 file_path_size - my_strlen(file_path) +
                     my_strlen(basename + 1));
    } else {
      my_strlcpy(file_path, file_name, file_path_size);
    }
  }
}

bool LinuxDumper::ReadAuxv() {
  char auxv_path[NAME_MAX];
  if (!BuildProcPath(auxv_path, pid_, "auxv")) {
    return false;
  }

  int fd = sys_open(auxv_path, O_RDONLY, 0);
  if (fd < 0) {
    return false;
  }

  elf_aux_entry one_aux_entry;
  bool res = false;
  while (sys_read(fd,
                  &one_aux_entry,
                  sizeof(elf_aux_entry)) == sizeof(elf_aux_entry) &&
         one_aux_entry.a_type != AT_NULL) {
    if (one_aux_entry.a_type <= AT_MAX) {
      auxv_[one_aux_entry.a_type] = one_aux_entry.a_un.a_val;
      res = true;
    }
  }
  sys_close(fd);
  return res;
}

bool LinuxDumper::EnumerateMappings() {
  char maps_path[NAME_MAX];
  if (!BuildProcPath(maps_path, pid_, "maps"))
    return false;

  // linux_gate_loc is the beginning of the kernel's mapping of
  // linux-gate.so in the process.  It doesn't actually show up in the
  // maps list as a filename, but it can be found using the AT_SYSINFO_EHDR
  // aux vector entry, which gives the information necessary to special
  // case its entry when creating the list of mappings.
  // See http://www.trilithium.com/johan/2005/08/linux-gate/ for more
  // information.
  const void* linux_gate_loc =
      reinterpret_cast<void*>(auxv_[AT_SYSINFO_EHDR]);
  // Although the initial executable is usually the first mapping, it's not
  // guaranteed (see http://crosbug.com/25355); therefore, try to use the
  // actual entry point to find the mapping.
  const void* entry_point_loc = reinterpret_cast<void*>(auxv_[AT_ENTRY]);

  const int fd = sys_open(maps_path, O_RDONLY, 0);
  if (fd < 0)
    return false;
  LineReader* const line_reader = new(allocator_) LineReader(fd);

  const char* line;
  unsigned line_len;
  while (line_reader->GetNextLine(&line, &line_len)) {
    uintptr_t start_addr, end_addr, offset;

    const char* i1 = my_read_hex_ptr(&start_addr, line);
    if (*i1 == '-') {
      const char* i2 = my_read_hex_ptr(&end_addr, i1 + 1);
      if (*i2 == ' ') {
        bool exec = (*(i2 + 3) == 'x');
        const char* i3 = my_read_hex_ptr(&offset, i2 + 6 /* skip ' rwxp ' */);
        if (*i3 == ' ') {
          const char* name = NULL;
          // Only copy name if the name is a valid path name, or if
          // it's the VDSO image.
          if (((name = my_strchr(line, '/')) == NULL) &&
              linux_gate_loc &&
              reinterpret_cast<void*>(start_addr) == linux_gate_loc) {
            name = kLinuxGateLibraryName;
            offset = 0;
          }
          // Merge adjacent mappings into one module, assuming they're a single
          // library mapped by the dynamic linker. Do this only if their name
          // matches and either they have the same +x protection flag, or if the
          // previous mapping is not executable and the new one is, to handle
          // lld's output (see crbug.com/716484).
          if (name && !mappings_.empty()) {
            MappingInfo* module = mappings_.back();
            if ((start_addr == module->start_addr + module->size) &&
                (my_strlen(name) == my_strlen(module->name)) &&
                (my_strncmp(name, module->name, my_strlen(name)) == 0) &&
                ((exec == module->exec) || (!module->exec && exec))) {
              module->system_mapping_info.end_addr = end_addr;
              module->size = end_addr - module->start_addr;
              module->exec |= exec;
              line_reader->PopLine(line_len);
              continue;
            }
          }
          MappingInfo* const module = new(allocator_) MappingInfo;
          mappings_.push_back(module);
          my_memset(module, 0, sizeof(MappingInfo));
          module->system_mapping_info.start_addr = start_addr;
          module->system_mapping_info.end_addr = end_addr;
          module->start_addr = start_addr;
          module->size = end_addr - start_addr;
          module->offset = offset;
          module->exec = exec;
          if (name != NULL) {
            const unsigned l = my_strlen(name);
            if (l < sizeof(module->name))
              my_memcpy(module->name, name, l);
          }
        }
      }
    }
    line_reader->PopLine(line_len);
  }

  if (entry_point_loc) {
    for (size_t i = 0; i < mappings_.size(); ++i) {
      MappingInfo* module = mappings_[i];

      // If this module contains the entry-point, and it's not already the first
      // one, then we need to make it be first.  This is because the minidump
      // format assumes the first module is the one that corresponds to the main
      // executable (as codified in
      // processor/minidump.cc:MinidumpModuleList::GetMainModule()).
      if ((entry_point_loc >= reinterpret_cast<void*>(module->start_addr)) &&
          (entry_point_loc <
           reinterpret_cast<void*>(module->start_addr + module->size))) {
        for (size_t j = i; j > 0; j--)
          mappings_[j] = mappings_[j - 1];
        mappings_[0] = module;
        break;
      }
    }
  }

  sys_close(fd);

  return !mappings_.empty();
}

#if defined(__ANDROID__)

bool LinuxDumper::GetLoadedElfHeader(uintptr_t start_addr, ElfW(Ehdr)* ehdr) {
  CopyFromProcess(ehdr, pid_,
                  reinterpret_cast<const void*>(start_addr),
                  sizeof(*ehdr));
  return my_memcmp(&ehdr->e_ident, ELFMAG, SELFMAG) == 0;
}

void LinuxDumper::ParseLoadedElfProgramHeaders(ElfW(Ehdr)* ehdr,
                                               uintptr_t start_addr,
                                               uintptr_t* min_vaddr_ptr,
                                               uintptr_t* dyn_vaddr_ptr,
                                               size_t* dyn_count_ptr) {
  uintptr_t phdr_addr = start_addr + ehdr->e_phoff;

  const uintptr_t max_addr = UINTPTR_MAX;
  uintptr_t min_vaddr = max_addr;
  uintptr_t dyn_vaddr = 0;
  size_t dyn_count = 0;

  for (size_t i = 0; i < ehdr->e_phnum; ++i) {
    ElfW(Phdr) phdr;
    CopyFromProcess(&phdr, pid_,
                    reinterpret_cast<const void*>(phdr_addr),
                    sizeof(phdr));
    if (phdr.p_type == PT_LOAD && phdr.p_vaddr < min_vaddr) {
      min_vaddr = phdr.p_vaddr;
    }
    if (phdr.p_type == PT_DYNAMIC) {
      dyn_vaddr = phdr.p_vaddr;
      dyn_count = phdr.p_memsz / sizeof(ElfW(Dyn));
    }
    phdr_addr += sizeof(phdr);
  }

  *min_vaddr_ptr = min_vaddr;
  *dyn_vaddr_ptr = dyn_vaddr;
  *dyn_count_ptr = dyn_count;
}

bool LinuxDumper::HasAndroidPackedRelocations(uintptr_t load_bias,
                                              uintptr_t dyn_vaddr,
                                              size_t dyn_count) {
  uintptr_t dyn_addr = load_bias + dyn_vaddr;
  for (size_t i = 0; i < dyn_count; ++i) {
    ElfW(Dyn) dyn;
    CopyFromProcess(&dyn, pid_,
                    reinterpret_cast<const void*>(dyn_addr),
                    sizeof(dyn));
    if (dyn.d_tag == DT_ANDROID_REL || dyn.d_tag == DT_ANDROID_RELA) {
      return true;
    }
    dyn_addr += sizeof(dyn);
  }
  return false;
}

uintptr_t LinuxDumper::GetEffectiveLoadBias(ElfW(Ehdr)* ehdr,
                                            uintptr_t start_addr) {
  uintptr_t min_vaddr = 0;
  uintptr_t dyn_vaddr = 0;
  size_t dyn_count = 0;
  ParseLoadedElfProgramHeaders(ehdr, start_addr,
                               &min_vaddr, &dyn_vaddr, &dyn_count);
  // If |min_vaddr| is non-zero and we find Android packed relocation tags,
  // return the effective load bias.
  if (min_vaddr != 0) {
    const uintptr_t load_bias = start_addr - min_vaddr;
    if (HasAndroidPackedRelocations(load_bias, dyn_vaddr, dyn_count)) {
      return load_bias;
    }
  }
  // Either |min_vaddr| is zero, or it is non-zero but we did not find the
  // expected Android packed relocations tags.
  return start_addr;
}

void LinuxDumper::LatePostprocessMappings() {
  for (size_t i = 0; i < mappings_.size(); ++i) {
    // Only consider exec mappings that indicate a file path was mapped, and
    // where the ELF header indicates a mapped shared library.
    MappingInfo* mapping = mappings_[i];
    if (!(mapping->exec && mapping->name[0] == '/')) {
      continue;
    }
    ElfW(Ehdr) ehdr;
    if (!GetLoadedElfHeader(mapping->start_addr, &ehdr)) {
      continue;
    }
    if (ehdr.e_type == ET_DYN) {
      // Compute the effective load bias for this mapped library, and update
      // the mapping to hold that rather than |start_addr|, at the same time
      // adjusting |size| to account for the change in |start_addr|. Where
      // the library does not contain Android packed relocations,
      // GetEffectiveLoadBias() returns |start_addr| and the mapping entry
      // is not changed.
      const uintptr_t load_bias = GetEffectiveLoadBias(&ehdr,
                                                       mapping->start_addr);
      mapping->size += mapping->start_addr - load_bias;
      mapping->start_addr = load_bias;
    }
  }
}

#endif  // __ANDROID__

// Get information about the stack, given the stack pointer. We don't try to
// walk the stack since we might not have all the information needed to do
// unwind. So we just grab, up to, 32k of stack.
bool LinuxDumper::GetStackInfo(const void** stack, size_t* stack_len,
                               uintptr_t int_stack_pointer) {
  // Move the stack pointer to the bottom of the page that it's in.
  const uintptr_t page_size = getpagesize();

  uint8_t* const stack_pointer =
      reinterpret_cast<uint8_t*>(int_stack_pointer & ~(page_size - 1));

  // The number of bytes of stack which we try to capture.
  static const ptrdiff_t kStackToCapture = 32 * 1024;

  const MappingInfo* mapping = FindMapping(stack_pointer);
  if (!mapping)
    return false;
  const ptrdiff_t offset = stack_pointer -
      reinterpret_cast<uint8_t*>(mapping->start_addr);
  const ptrdiff_t distance_to_end =
      static_cast<ptrdiff_t>(mapping->size) - offset;
  *stack_len = distance_to_end > kStackToCapture ?
      kStackToCapture : distance_to_end;
  *stack = stack_pointer;
  return true;
}

void LinuxDumper::SanitizeStackCopy(uint8_t* stack_copy, size_t stack_len,
                                    uintptr_t stack_pointer,
                                    uintptr_t sp_offset) {
  // We optimize the search for containing mappings in three ways:
  // 1) We expect that pointers into the stack mapping will be common, so
  //    we cache that address range.
  // 2) The last referenced mapping is a reasonable predictor for the next
  //    referenced mapping, so we test that first.
  // 3) We precompute a bitfield based upon bits 32:32-n of the start and
  //    stop addresses, and use that to short circuit any values that can
  //    not be pointers. (n=11)
  const uintptr_t defaced =
#if defined(__LP64__)
      0x0defaced0defaced;
#else
      0x0defaced;
#endif
  // the bitfield length is 2^test_bits long.
  const unsigned int test_bits = 11;
  // byte length of the corresponding array.
  const unsigned int array_size = 1 << (test_bits - 3);
  const unsigned int array_mask = array_size - 1;
  // The amount to right shift pointers by. This captures the top bits
  // on 32 bit architectures. On 64 bit architectures this would be
  // uninformative so we take the same range of bits.
  const unsigned int shift = 32 - 11;
  const MappingInfo* last_hit_mapping = nullptr;
  const MappingInfo* hit_mapping = nullptr;
  const MappingInfo* stack_mapping = FindMappingNoBias(stack_pointer);
  // The magnitude below which integers are considered to be to be
  // 'small', and not constitute a PII risk. These are included to
  // avoid eliding useful register values.
  const ssize_t small_int_magnitude = 4096;

  char could_hit_mapping[array_size];
  my_memset(could_hit_mapping, 0, array_size);

  // Initialize the bitfield such that if the (pointer >> shift)'th
  // bit, modulo the bitfield size, is not set then there does not
  // exist a mapping in mappings_ that would contain that pointer.
  for (size_t i = 0; i < mappings_.size(); ++i) {
    if (!mappings_[i]->exec) continue;
    // For each mapping, work out the (unmodulo'ed) range of bits to
    // set.
    uintptr_t start = mappings_[i]->start_addr;
    uintptr_t end = start + mappings_[i]->size;
    start >>= shift;
    end >>= shift;
    for (size_t bit = start; bit <= end; ++bit) {
      // Set each bit in the range, applying the modulus.
      could_hit_mapping[(bit >> 3) & array_mask] |= 1 << (bit & 7);
    }
  }

  // Zero memory that is below the current stack pointer.
  const uintptr_t offset =
      (sp_offset + sizeof(uintptr_t) - 1) & ~(sizeof(uintptr_t) - 1);
  if (offset) {
    my_memset(stack_copy, 0, offset);
  }

  // Apply sanitization to each complete pointer-aligned word in the
  // stack.
  uint8_t* sp;
  for (sp = stack_copy + offset;
       sp <= stack_copy + stack_len - sizeof(uintptr_t);
       sp += sizeof(uintptr_t)) {
    uintptr_t addr;
    my_memcpy(&addr, sp, sizeof(uintptr_t));
    if (static_cast<intptr_t>(addr) <= small_int_magnitude &&
        static_cast<intptr_t>(addr) >= -small_int_magnitude) {
      continue;
    }
    if (stack_mapping && MappingContainsAddress(*stack_mapping, addr)) {
      continue;
    }
    if (last_hit_mapping && MappingContainsAddress(*last_hit_mapping, addr)) {
      continue;
    }
    uintptr_t test = addr >> shift;
    if (could_hit_mapping[(test >> 3) & array_mask] & (1 << (test & 7)) &&
        (hit_mapping = FindMappingNoBias(addr)) != nullptr &&
        hit_mapping->exec) {
      last_hit_mapping = hit_mapping;
      continue;
    }
    my_memcpy(sp, &defaced, sizeof(uintptr_t));
  }
  // Zero any partial word at the top of the stack, if alignment is
  // such that that is required.
  if (sp < stack_copy + stack_len) {
    my_memset(sp, 0, stack_copy + stack_len - sp);
  }
}

bool LinuxDumper::StackHasPointerToMapping(const uint8_t* stack_copy,
                                           size_t stack_len,
                                           uintptr_t sp_offset,
                                           const MappingInfo& mapping) {
  // Loop over all stack words that would have been on the stack in
  // the target process (i.e. are word aligned, and at addresses >=
  // the stack pointer).  Regardless of the alignment of |stack_copy|,
  // the memory starting at |stack_copy| + |offset| represents an
  // aligned word in the target process.
  const uintptr_t low_addr = mapping.system_mapping_info.start_addr;
  const uintptr_t high_addr = mapping.system_mapping_info.end_addr;
  const uintptr_t offset =
      (sp_offset + sizeof(uintptr_t) - 1) & ~(sizeof(uintptr_t) - 1);

  for (const uint8_t* sp = stack_copy + offset;
       sp <= stack_copy + stack_len - sizeof(uintptr_t);
       sp += sizeof(uintptr_t)) {
    uintptr_t addr;
    my_memcpy(&addr, sp, sizeof(uintptr_t));
    if (low_addr <= addr && addr <= high_addr)
      return true;
  }
  return false;
}

// Find the mapping which the given memory address falls in.
const MappingInfo* LinuxDumper::FindMapping(const void* address) const {
  const uintptr_t addr = (uintptr_t) address;

  for (size_t i = 0; i < mappings_.size(); ++i) {
    const uintptr_t start = static_cast<uintptr_t>(mappings_[i]->start_addr);
    if (addr >= start && addr - start < mappings_[i]->size)
      return mappings_[i];
  }

  return NULL;
}

// Find the mapping which the given memory address falls in. Uses the
// unadjusted mapping address range from the kernel, rather than the
// biased range.
const MappingInfo* LinuxDumper::FindMappingNoBias(uintptr_t address) const {
  for (size_t i = 0; i < mappings_.size(); ++i) {
    if (address >= mappings_[i]->system_mapping_info.start_addr &&
        address < mappings_[i]->system_mapping_info.end_addr) {
      return mappings_[i];
    }
  }
  return NULL;
}

bool LinuxDumper::HandleDeletedFileInMapping(char* path) const {
  static const size_t kDeletedSuffixLen = sizeof(kDeletedSuffix) - 1;

  // Check for ' (deleted)' in |path|.
  // |path| has to be at least as long as "/x (deleted)".
  const size_t path_len = my_strlen(path);
  if (path_len < kDeletedSuffixLen + 2)
    return false;
  if (my_strncmp(path + path_len - kDeletedSuffixLen, kDeletedSuffix,
                 kDeletedSuffixLen) != 0) {
    return false;
  }

  // Check |path| against the /proc/pid/exe 'symlink'.
  char exe_link[NAME_MAX];
  if (!BuildProcPath(exe_link, pid_, "exe"))
    return false;
  MappingInfo new_mapping = {0};
  if (!SafeReadLink(exe_link, new_mapping.name))
    return false;
  char new_path[PATH_MAX];
  if (!GetMappingAbsolutePath(new_mapping, new_path))
    return false;
  if (my_strcmp(path, new_path) != 0)
    return false;

  // Check to see if someone actually named their executable 'foo (deleted)'.
  struct kernel_stat exe_stat;
  struct kernel_stat new_path_stat;
  if (sys_stat(exe_link, &exe_stat) == 0 &&
      sys_stat(new_path, &new_path_stat) == 0 &&
      exe_stat.st_dev == new_path_stat.st_dev &&
      exe_stat.st_ino == new_path_stat.st_ino) {
    return false;
  }

  my_memcpy(path, exe_link, NAME_MAX);
  return true;
}

}  // namespace google_breakpad
