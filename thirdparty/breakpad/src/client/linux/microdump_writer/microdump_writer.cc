// Copyright (c) 2014, Google Inc.
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

// This translation unit generates microdumps into the console (logcat on
// Android). See crbug.com/410294 for more info and design docs.

#include "client/linux/microdump_writer/microdump_writer.h"

#include <limits>

#include <sys/utsname.h>

#include "client/linux/dump_writer_common/thread_info.h"
#include "client/linux/dump_writer_common/ucontext_reader.h"
#include "client/linux/handler/exception_handler.h"
#include "client/linux/handler/microdump_extra_info.h"
#include "client/linux/log/log.h"
#include "client/linux/minidump_writer/linux_ptrace_dumper.h"
#include "common/linux/file_id.h"
#include "common/linux/linux_libc_support.h"
#include "common/memory_allocator.h"

namespace {

using google_breakpad::auto_wasteful_vector;
using google_breakpad::ExceptionHandler;
using google_breakpad::kDefaultBuildIdSize;
using google_breakpad::LinuxDumper;
using google_breakpad::LinuxPtraceDumper;
using google_breakpad::MappingInfo;
using google_breakpad::MappingList;
using google_breakpad::MicrodumpExtraInfo;
using google_breakpad::RawContextCPU;
using google_breakpad::ThreadInfo;
using google_breakpad::UContextReader;

const size_t kLineBufferSize = 2048;

#if !defined(__LP64__)
// The following are only used by DumpFreeSpace, so need to be compiled
// in conditionally in the same way.

template <typename Dst, typename Src>
Dst saturated_cast(Src src) {
  if (src >= std::numeric_limits<Dst>::max())
    return std::numeric_limits<Dst>::max();
  if (src <= std::numeric_limits<Dst>::min())
    return std::numeric_limits<Dst>::min();
  return static_cast<Dst>(src);
}

int Log2Floor(uint64_t n) {
  // Copied from chromium src/base/bits.h
  if (n == 0)
    return -1;
  int log = 0;
  uint64_t value = n;
  for (int i = 5; i >= 0; --i) {
    int shift = (1 << i);
    uint64_t x = value >> shift;
    if (x != 0) {
      value = x;
      log += shift;
    }
  }
  assert(value == 1u);
  return log;
}

bool MappingsAreAdjacent(const MappingInfo& a, const MappingInfo& b) {
  // Because of load biasing, we can end up with a situation where two
  // mappings actually overlap. So we will define adjacency to also include a
  // b start address that lies within a's address range (including starting
  // immediately after a).
  // Because load biasing only ever moves the start address backwards, the end
  // address should still increase.
  return a.start_addr <= b.start_addr && a.start_addr + a.size >= b.start_addr;
}

bool MappingLessThan(const MappingInfo* a, const MappingInfo* b) {
  // Return true if mapping a is before mapping b.
  // For the same reason (load biasing) we compare end addresses, which - unlike
  // start addresses - will not have been modified.
  return a->start_addr + a->size < b->start_addr + b->size;
}

size_t NextOrderedMapping(
    const google_breakpad::wasteful_vector<MappingInfo*>& mappings,
    size_t curr) {
  // Find the mapping that directly follows mappings[curr].
  // If no such mapping exists, return |invalid| to indicate this.
  const size_t invalid = std::numeric_limits<size_t>::max();
  size_t best = invalid;
  for (size_t next = 0; next < mappings.size(); ++next) {
    if (MappingLessThan(mappings[curr], mappings[next]) &&
        (best == invalid || MappingLessThan(mappings[next], mappings[best]))) {
      best = next;
    }
  }
  return best;
}

#endif  // !__LP64__

class MicrodumpWriter {
 public:
  MicrodumpWriter(const ExceptionHandler::CrashContext* context,
                  const MappingList& mappings,
                  bool skip_dump_if_principal_mapping_not_referenced,
                  uintptr_t address_within_principal_mapping,
                  bool sanitize_stack,
                  const MicrodumpExtraInfo& microdump_extra_info,
                  LinuxDumper* dumper)
      : ucontext_(context ? &context->context : NULL),
#if !defined(__ARM_EABI__) && !defined(__mips__)
        float_state_(context ? &context->float_state : NULL),
#endif
        dumper_(dumper),
        mapping_list_(mappings),
        skip_dump_if_principal_mapping_not_referenced_(
            skip_dump_if_principal_mapping_not_referenced),
        address_within_principal_mapping_(address_within_principal_mapping),
        sanitize_stack_(sanitize_stack),
        microdump_extra_info_(microdump_extra_info),
        log_line_(NULL),
        stack_copy_(NULL),
        stack_len_(0),
        stack_lower_bound_(0),
        stack_pointer_(0) {
    log_line_ = reinterpret_cast<char*>(Alloc(kLineBufferSize));
    if (log_line_)
      log_line_[0] = '\0';  // Clear out the log line buffer.
  }

  ~MicrodumpWriter() { dumper_->ThreadsResume(); }

  bool Init() {
    // In the exceptional case where the system was out of memory and there
    // wasn't even room to allocate the line buffer, bail out. There is nothing
    // useful we can possibly achieve without the ability to Log. At least let's
    // try to not crash.
    if (!dumper_->Init() || !log_line_)
      return false;
    return dumper_->ThreadsSuspend() && dumper_->LateInit();
  }

  void Dump() {
    CaptureResult stack_capture_result = CaptureCrashingThreadStack(-1);
    if (stack_capture_result == CAPTURE_UNINTERESTING) {
      LogLine("Microdump skipped (uninteresting)");
      return;
    }

    LogLine("-----BEGIN BREAKPAD MICRODUMP-----");
    DumpProductInformation();
    DumpOSInformation();
    DumpProcessType();
    DumpCrashReason();
    DumpGPUInformation();
#if !defined(__LP64__)
    DumpFreeSpace();
#endif
    if (stack_capture_result == CAPTURE_OK)
      DumpThreadStack();
    DumpCPUState();
    DumpMappings();
    LogLine("-----END BREAKPAD MICRODUMP-----");
  }

 private:
  enum CaptureResult { CAPTURE_OK, CAPTURE_FAILED, CAPTURE_UNINTERESTING };

  // Writes one line to the system log.
  void LogLine(const char* msg) {
#if defined(__ANDROID__)
    logger::writeToCrashLog(msg);
#else
    logger::write(msg, my_strlen(msg));
    logger::write("\n", 1);
#endif
  }

  // Stages the given string in the current line buffer.
  void LogAppend(const char* str) {
    my_strlcat(log_line_, str, kLineBufferSize);
  }

  // As above (required to take precedence over template specialization below).
  void LogAppend(char* str) {
    LogAppend(const_cast<const char*>(str));
  }

  // Stages the hex repr. of the given int type in the current line buffer.
  template<typename T>
  void LogAppend(T value) {
    // Make enough room to hex encode the largest int type + NUL.
    static const char HEX[] = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                               'A', 'B', 'C', 'D', 'E', 'F'};
    char hexstr[sizeof(T) * 2 + 1];
    for (int i = sizeof(T) * 2 - 1; i >= 0; --i, value >>= 4)
      hexstr[i] = HEX[static_cast<uint8_t>(value) & 0x0F];
    hexstr[sizeof(T) * 2] = '\0';
    LogAppend(hexstr);
  }

  // Stages the buffer content hex-encoded in the current line buffer.
  void LogAppend(const void* buf, size_t length) {
    const uint8_t* ptr = reinterpret_cast<const uint8_t*>(buf);
    for (size_t i = 0; i < length; ++i, ++ptr)
      LogAppend(*ptr);
  }

  // Writes out the current line buffer on the system log.
  void LogCommitLine() {
    LogLine(log_line_);
    log_line_[0] = 0;
  }

  CaptureResult CaptureCrashingThreadStack(int max_stack_len) {
    stack_pointer_ = UContextReader::GetStackPointer(ucontext_);

    if (!dumper_->GetStackInfo(reinterpret_cast<const void**>(&stack_lower_bound_),
                               &stack_len_, stack_pointer_)) {
      return CAPTURE_FAILED;
    }

    if (max_stack_len >= 0 &&
        stack_len_ > static_cast<size_t>(max_stack_len)) {
      stack_len_ = max_stack_len;
    }

    stack_copy_ = reinterpret_cast<uint8_t*>(Alloc(stack_len_));
    dumper_->CopyFromProcess(stack_copy_, dumper_->crash_thread(),
                             reinterpret_cast<const void*>(stack_lower_bound_),
                             stack_len_);

    if (!skip_dump_if_principal_mapping_not_referenced_) return CAPTURE_OK;

    const MappingInfo* principal_mapping =
        dumper_->FindMappingNoBias(address_within_principal_mapping_);
    if (!principal_mapping) return CAPTURE_UNINTERESTING;

    uintptr_t low_addr = principal_mapping->system_mapping_info.start_addr;
    uintptr_t high_addr = principal_mapping->system_mapping_info.end_addr;
    uintptr_t pc = UContextReader::GetInstructionPointer(ucontext_);
    if (low_addr <= pc && pc <= high_addr) return CAPTURE_OK;

    if (dumper_->StackHasPointerToMapping(stack_copy_, stack_len_,
                                          stack_pointer_ - stack_lower_bound_,
                                          *principal_mapping)) {
      return CAPTURE_OK;
    }
    return CAPTURE_UNINTERESTING;
  }

  void DumpProductInformation() {
    LogAppend("V ");
    if (microdump_extra_info_.product_info) {
      LogAppend(microdump_extra_info_.product_info);
    } else {
      LogAppend("UNKNOWN:0.0.0.0");
    }
    LogCommitLine();
  }

  void DumpProcessType() {
    LogAppend("P ");
    if (microdump_extra_info_.process_type) {
      LogAppend(microdump_extra_info_.process_type);
    } else {
      LogAppend("UNKNOWN");
    }
    LogCommitLine();
  }

  void DumpCrashReason() {
    LogAppend("R ");
    LogAppend(dumper_->crash_signal());
    LogAppend(" ");
    LogAppend(dumper_->GetCrashSignalString());
    LogAppend(" ");
    LogAppend(dumper_->crash_address());
    LogCommitLine();
  }

  void DumpOSInformation() {
    const uint8_t n_cpus = static_cast<uint8_t>(sysconf(_SC_NPROCESSORS_CONF));

#if defined(__ANDROID__)
    const char kOSId[] = "A";
#else
    const char kOSId[] = "L";
#endif

// Dump the runtime architecture. On multiarch devices it might not match the
// hw architecture (the one returned by uname()), for instance in the case of
// a 32-bit app running on a aarch64 device.
#if defined(__aarch64__)
    const char kArch[] = "arm64";
#elif defined(__ARMEL__)
    const char kArch[] = "arm";
#elif defined(__x86_64__)
    const char kArch[] = "x86_64";
#elif defined(__i386__)
    const char kArch[] = "x86";
#elif defined(__mips__)
# if _MIPS_SIM == _ABIO32
    const char kArch[] = "mips";
# elif _MIPS_SIM == _ABI64
    const char kArch[] = "mips64";
# else
#  error "This mips ABI is currently not supported (n32)"
#endif
#else
#error "This code has not been ported to your platform yet"
#endif

    LogAppend("O ");
    LogAppend(kOSId);
    LogAppend(" ");
    LogAppend(kArch);
    LogAppend(" ");
    LogAppend(n_cpus);
    LogAppend(" ");

    // Dump the HW architecture (e.g., armv7l, aarch64).
    struct utsname uts;
    const bool has_uts_info = (uname(&uts) == 0);
    const char* hwArch = has_uts_info ? uts.machine : "unknown_hw_arch";
    LogAppend(hwArch);
    LogAppend(" ");

    // If the client has attached a build fingerprint to the MinidumpDescriptor
    // use that one. Otherwise try to get some basic info from uname().
    if (microdump_extra_info_.build_fingerprint) {
      LogAppend(microdump_extra_info_.build_fingerprint);
    } else if (has_uts_info) {
      LogAppend(uts.release);
      LogAppend(" ");
      LogAppend(uts.version);
    } else {
      LogAppend("no build fingerprint available");
    }
    LogCommitLine();
  }

  void DumpGPUInformation() {
    LogAppend("G ");
    if (microdump_extra_info_.gpu_fingerprint) {
      LogAppend(microdump_extra_info_.gpu_fingerprint);
    } else {
      LogAppend("UNKNOWN");
    }
    LogCommitLine();
  }

  void DumpThreadStack() {
    if (sanitize_stack_) {
      dumper_->SanitizeStackCopy(stack_copy_, stack_len_, stack_pointer_,
                                 stack_pointer_ - stack_lower_bound_);
    }

    LogAppend("S 0 ");
    LogAppend(stack_pointer_);
    LogAppend(" ");
    LogAppend(stack_lower_bound_);
    LogAppend(" ");
    LogAppend(stack_len_);
    LogCommitLine();

    const size_t STACK_DUMP_CHUNK_SIZE = 384;
    for (size_t stack_off = 0; stack_off < stack_len_;
         stack_off += STACK_DUMP_CHUNK_SIZE) {
      LogAppend("S ");
      LogAppend(stack_lower_bound_ + stack_off);
      LogAppend(" ");
      LogAppend(stack_copy_ + stack_off,
                std::min(STACK_DUMP_CHUNK_SIZE, stack_len_ - stack_off));
      LogCommitLine();
    }
  }

  void DumpCPUState() {
    RawContextCPU cpu;
    my_memset(&cpu, 0, sizeof(RawContextCPU));
#if !defined(__ARM_EABI__) && !defined(__mips__)
    UContextReader::FillCPUContext(&cpu, ucontext_, float_state_);
#else
    UContextReader::FillCPUContext(&cpu, ucontext_);
#endif
    LogAppend("C ");
    LogAppend(&cpu, sizeof(cpu));
    LogCommitLine();
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

  // Dump information about the provided |mapping|. If |identifier| is non-NULL,
  // use it instead of calculating a file ID from the mapping.
  void DumpModule(const MappingInfo& mapping,
                  bool member,
                  unsigned int mapping_id,
                  const uint8_t* identifier) {

    auto_wasteful_vector<uint8_t, kDefaultBuildIdSize> identifier_bytes(
        dumper_->allocator());

    if (identifier) {
      // GUID was provided by caller.
      identifier_bytes.insert(identifier_bytes.end(),
                              identifier,
                              identifier + sizeof(MDGUID));
    } else {
      dumper_->ElfFileIdentifierForMapping(
          mapping,
          member,
          mapping_id,
          identifier_bytes);
    }

    // Copy as many bytes of |identifier| as will fit into a MDGUID
    MDGUID module_identifier = {0};
    memcpy(&module_identifier, &identifier_bytes[0],
           std::min(sizeof(MDGUID), identifier_bytes.size()));

    char file_name[NAME_MAX];
    char file_path[NAME_MAX];
    dumper_->GetMappingEffectiveNameAndPath(
        mapping, file_path, sizeof(file_path), file_name, sizeof(file_name));

    LogAppend("M ");
    LogAppend(static_cast<uintptr_t>(mapping.start_addr));
    LogAppend(" ");
    LogAppend(mapping.offset);
    LogAppend(" ");
    LogAppend(mapping.size);
    LogAppend(" ");
    LogAppend(module_identifier.data1);
    LogAppend(module_identifier.data2);
    LogAppend(module_identifier.data3);
    LogAppend(module_identifier.data4[0]);
    LogAppend(module_identifier.data4[1]);
    LogAppend(module_identifier.data4[2]);
    LogAppend(module_identifier.data4[3]);
    LogAppend(module_identifier.data4[4]);
    LogAppend(module_identifier.data4[5]);
    LogAppend(module_identifier.data4[6]);
    LogAppend(module_identifier.data4[7]);
    LogAppend("0 ");  // Age is always 0 on Linux.
    LogAppend(file_name);
    LogCommitLine();
  }

#if !defined(__LP64__)
  void DumpFreeSpace() {
    const MappingInfo* stack_mapping = nullptr;
    ThreadInfo info;
    if (dumper_->GetThreadInfoByIndex(dumper_->GetMainThreadIndex(), &info)) {
      stack_mapping = dumper_->FindMappingNoBias(info.stack_pointer);
    }

    const google_breakpad::wasteful_vector<MappingInfo*>& mappings =
        dumper_->mappings();
    if (mappings.size() == 0) return;

    // This is complicated by the fact that mappings is not in order. It should
    // be mostly in order, however the mapping that contains the entry point for
    // the process is always at the front of the vector.

    static const int HBITS = sizeof(size_t) * 8;
    size_t hole_histogram[HBITS];
    my_memset(hole_histogram, 0, sizeof(hole_histogram));

    // Find the lowest address mapping.
    size_t curr = 0;
    for (size_t i = 1; i < mappings.size(); ++i) {
      if (mappings[i]->start_addr < mappings[curr]->start_addr) curr = i;
    }

    uintptr_t lo_addr = mappings[curr]->start_addr;

    size_t hole_cnt = 0;
    size_t hole_max = 0;
    size_t hole_sum = 0;

    while (true) {
      // Skip to the end of an adjacent run of mappings. This is an optimization
      // for the fact that mappings is mostly sorted.
      while (curr != mappings.size() - 1 &&
             MappingsAreAdjacent(*mappings[curr], *mappings[curr + 1])) {
        ++curr;
      }

      if (mappings[curr] == stack_mapping) {
        // Because we can't determine the top of userspace mappable
        // memory we treat the start of the process stack as the top
        // of the allocatable address space. Once we reach
        // |stack_mapping| we are done scanning for free space regions.
        break;
      }

      size_t next = NextOrderedMapping(mappings, curr);
      if (next == std::numeric_limits<size_t>::max())
        break;

      uintptr_t hole_lo = mappings[curr]->start_addr + mappings[curr]->size;
      uintptr_t hole_hi = mappings[next]->start_addr;

      if (hole_hi > hole_lo) {
        size_t hole_sz = hole_hi - hole_lo;
        hole_sum += hole_sz;
        hole_max = std::max(hole_sz, hole_max);
        ++hole_cnt;
        ++hole_histogram[Log2Floor(hole_sz)];
      }
      curr = next;
    }

    uintptr_t hi_addr = mappings[curr]->start_addr + mappings[curr]->size;

    LogAppend("H ");
    LogAppend(lo_addr);
    LogAppend(" ");
    LogAppend(hi_addr);
    LogAppend(" ");
    LogAppend(saturated_cast<uint16_t>(hole_cnt));
    LogAppend(" ");
    LogAppend(hole_max);
    LogAppend(" ");
    LogAppend(hole_sum);
    for (unsigned int i = 0; i < HBITS; ++i) {
      if (!hole_histogram[i]) continue;
      LogAppend(" ");
      LogAppend(saturated_cast<uint8_t>(i));
      LogAppend(":");
      LogAppend(saturated_cast<uint8_t>(hole_histogram[i]));
    }
    LogCommitLine();
  }
#endif

  // Write information about the mappings in effect.
  void DumpMappings() {
    // First write all the mappings from the dumper
    for (unsigned i = 0; i < dumper_->mappings().size(); ++i) {
      const MappingInfo& mapping = *dumper_->mappings()[i];
      if (mapping.name[0] == 0 ||  // only want modules with filenames.
          !mapping.exec ||  // only want executable mappings.
          mapping.size < 4096 || // too small to get a signature for.
          HaveMappingInfo(mapping)) {
        continue;
      }

      DumpModule(mapping, true, i, NULL);
    }
    // Next write all the mappings provided by the caller
    for (MappingList::const_iterator iter = mapping_list_.begin();
         iter != mapping_list_.end();
         ++iter) {
      DumpModule(iter->first, false, 0, iter->second);
    }
  }

  void* Alloc(unsigned bytes) { return dumper_->allocator()->Alloc(bytes); }

  const ucontext_t* const ucontext_;
#if !defined(__ARM_EABI__) && !defined(__mips__)
  const google_breakpad::fpstate_t* const float_state_;
#endif
  LinuxDumper* dumper_;
  const MappingList& mapping_list_;
  bool skip_dump_if_principal_mapping_not_referenced_;
  uintptr_t address_within_principal_mapping_;
  bool sanitize_stack_;
  const MicrodumpExtraInfo microdump_extra_info_;
  char* log_line_;

  // The local copy of crashed process stack memory, beginning at
  // |stack_lower_bound_|.
  uint8_t* stack_copy_;

  // The length of crashed process stack copy.
  size_t stack_len_;

  // The address of the page containing the stack pointer in the
  // crashed process. |stack_lower_bound_| <= |stack_pointer_|
  uintptr_t stack_lower_bound_;

  // The stack pointer of the crashed thread.
  uintptr_t stack_pointer_;
};
}  // namespace

namespace google_breakpad {

bool WriteMicrodump(pid_t crashing_process,
                    const void* blob,
                    size_t blob_size,
                    const MappingList& mappings,
                    bool skip_dump_if_principal_mapping_not_referenced,
                    uintptr_t address_within_principal_mapping,
                    bool sanitize_stack,
                    const MicrodumpExtraInfo& microdump_extra_info) {
  LinuxPtraceDumper dumper(crashing_process);
  const ExceptionHandler::CrashContext* context = NULL;
  if (blob) {
    if (blob_size != sizeof(ExceptionHandler::CrashContext))
      return false;
    context = reinterpret_cast<const ExceptionHandler::CrashContext*>(blob);
    dumper.SetCrashInfoFromSigInfo(context->siginfo);
    dumper.set_crash_thread(context->tid);
  }
  MicrodumpWriter writer(context, mappings,
                         skip_dump_if_principal_mapping_not_referenced,
                         address_within_principal_mapping, sanitize_stack,
                         microdump_extra_info, &dumper);
  if (!writer.Init())
    return false;
  writer.Dump();
  return true;
}

}  // namespace google_breakpad
