// Copyright 2007 Google LLC
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

// Author: Alfred Peng

#ifdef HAVE_CONFIG_H
#include <config.h>  // Must come first
#endif

#include <fcntl.h>
#include <sys/frame.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/utsname.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdlib>
#include <ctime>

#include "client/solaris/handler/minidump_generator.h"
#include "client/minidump_file_writer-inl.h"
#include "common/solaris/file_id.h"

namespace {

using namespace google_breakpad;
using namespace google_breakpad::elf::FileID;

// Argument for the writer function.
struct WriterArgument {
  MinidumpFileWriter* minidump_writer;

  // Pid of the lwp who called WriteMinidumpToFile
  int requester_pid;

  // The stack bottom of the lwp which caused the dump.
  // Mainly used to find the lwp id of the crashed lwp since signal
  // handler may not be called in the lwp who caused it.
  uintptr_t crashed_stack_bottom;

  // Id of the crashing lwp.
  int crashed_lwpid;

  // Signal number when crash happened. Can be 0 if this is a requested dump.
  int signo;

  // The ebp of the signal handler frame on x86.  Can be 0 if this is a
  // requested dump.
  uintptr_t sighandler_ebp;

  // User context when crash happens. Can be NULL if this is a requested dump.
  // This is actually an out parameter, but it will be filled in at the start
  // of the writer LWP.
  ucontext_t* sig_ctx;

  // Used to get information about the lwps.
  SolarisLwp* lwp_lister;
};

// Holding context information for the callback of finding the crashing lwp.
struct FindCrashLwpContext {
  const SolarisLwp* lwp_lister;
  uintptr_t crashing_stack_bottom;
  int crashing_lwpid;

  FindCrashLwpContext() :
    lwp_lister(NULL),
    crashing_stack_bottom(0UL),
    crashing_lwpid(-1) {
  }
};

// Callback for list lwps.
// It will compare the stack bottom of the provided lwp with the stack
// bottom of the crashed lwp, it they are eqaul, this lwp is the one
// who crashed.
bool IsLwpCrashedCallback(lwpstatus_t* lsp, void* context) {
  FindCrashLwpContext* crashing_context =
    static_cast<FindCrashLwpContext*>(context);
  const SolarisLwp* lwp_lister = crashing_context->lwp_lister;
  const prgregset_t* gregs = &(lsp->pr_reg);
#if TARGET_CPU_SPARC
  uintptr_t last_ebp = (*gregs)[R_FP];
#elif TARGET_CPU_X86
  uintptr_t last_ebp = (*gregs)[EBP];
#endif
  uintptr_t stack_bottom = lwp_lister->GetLwpStackBottom(last_ebp);
  if (stack_bottom > last_ebp &&
      stack_bottom == crashing_context->crashing_stack_bottom) {
    // Got it. Stop iteration.
    crashing_context->crashing_lwpid = lsp->pr_lwpid;
    return false;
  }

  return true;
}

// Find the crashing lwpid.
// This is done based on stack bottom comparing.
int FindCrashingLwp(uintptr_t crashing_stack_bottom,
                    int requester_pid,
                    const SolarisLwp* lwp_lister) {
  FindCrashLwpContext context;
  context.lwp_lister = lwp_lister;
  context.crashing_stack_bottom = crashing_stack_bottom;
  CallbackParam<LwpCallback> callback_param(IsLwpCrashedCallback,
                                            &context);
  lwp_lister->Lwp_iter_all(lwp_lister->getpid(), &callback_param);
  return context.crashing_lwpid;
}

bool WriteLwpStack(const SolarisLwp* lwp_lister,
                   uintptr_t last_esp,
                   UntypedMDRVA* memory,
                   MDMemoryDescriptor* loc) {
  uintptr_t stack_bottom = lwp_lister->GetLwpStackBottom(last_esp);
  if (stack_bottom >= last_esp) {
    int size = stack_bottom - last_esp;
    if (size > 0) {
      if (!memory->Allocate(size))
        return false;
      memory->Copy(reinterpret_cast<void*>(last_esp), size);
      loc->start_of_memory_range = last_esp;
      loc->memory = memory->location();
    }
    return true;
  }
  return false;
}

#if TARGET_CPU_SPARC
bool WriteContext(MDRawContextSPARC* context, ucontext_t* sig_ctx) {
  assert(sig_ctx != NULL);
  int* regs = sig_ctx->uc_mcontext.gregs;
  context->context_flags = MD_CONTEXT_SPARC_FULL;

  context->ccr = (unsigned int)(regs[0]);
  context->pc = (unsigned int)(regs[REG_PC]);
  context->npc = (unsigned int)(regs[REG_nPC]);
  context->y = (unsigned int)(regs[REG_Y]);
  context->asi = (unsigned int)(regs[19]);
  context->fprs = (unsigned int)(regs[20]);

  for ( int i = 0 ; i < 32; ++i ) {
    context->g_r[i] = 0;
  }

  for ( int i = 1 ; i < 16; ++i ) {
    context->g_r[i] = (uintptr_t)(sig_ctx->uc_mcontext.gregs[i + 3]);
  }
  context->g_r[30] = (uintptr_t)(((struct frame*)context->g_r[14])->fr_savfp);

  return true;
}

bool WriteContext(MDRawContextSPARC* context, prgregset_t regs,
                  prfpregset_t* fp_regs) {
  if (!context || !regs)
    return false;

  context->context_flags = MD_CONTEXT_SPARC_FULL;

  context->ccr = (uintptr_t)(regs[32]);
  context->pc = (uintptr_t)(regs[R_PC]);
  context->npc = (uintptr_t)(regs[R_nPC]);
  context->y = (uintptr_t)(regs[R_Y]);
  context->asi = (uintptr_t)(regs[36]);
  context->fprs = (uintptr_t)(regs[37]);
  for ( int i = 0 ; i < 32 ; ++i ){
    context->g_r[i] = (uintptr_t)(regs[i]);
  }

  return true;
}
#elif TARGET_CPU_X86
bool WriteContext(MDRawContextX86* context, prgregset_t regs,
                  prfpregset_t* fp_regs) {
  if (!context || !regs)
    return false;

  context->context_flags = MD_CONTEXT_X86_FULL;

  context->cs = regs[CS];
  context->ds = regs[DS];
  context->es = regs[ES];
  context->fs = regs[FS];
  context->gs = regs[GS];
  context->ss = regs[SS];
  context->edi = regs[EDI];
  context->esi = regs[ESI];
  context->ebx = regs[EBX];
  context->edx = regs[EDX];
  context->ecx = regs[ECX];
  context->eax = regs[EAX];
  context->ebp = regs[EBP];
  context->eip = regs[EIP];
  context->esp = regs[UESP];
  context->eflags = regs[EFL];

  return true;
}
#endif /* TARGET_CPU_XXX */

// Write information about a crashed Lwp.
// When a lwp crash, kernel will write something on the stack for processing
// signal. This makes the current stack not reliable, and our stack walker
// won't figure out the whole call stack for this. So we write the stack at the
// time of the crash into the minidump file, not the current stack.
bool WriteCrashedLwpStream(MinidumpFileWriter* minidump_writer,
                           const WriterArgument* writer_args,
                           const lwpstatus_t* lsp,
                           MDRawThread* lwp) {
  assert(writer_args->sig_ctx != NULL);

  lwp->thread_id = lsp->pr_lwpid;

#if TARGET_CPU_SPARC
  UntypedMDRVA memory(minidump_writer);
  if (!WriteLwpStack(writer_args->lwp_lister,
                     writer_args->sig_ctx->uc_mcontext.gregs[REG_O6],
                     &memory,
                     &lwp->stack))
    return false;

  TypedMDRVA<MDRawContextSPARC> context(minidump_writer);
  if (!context.Allocate())
    return false;
  lwp->thread_context = context.location();
  memset(context.get(), 0, sizeof(MDRawContextSPARC));
  return WriteContext(context.get(), writer_args->sig_ctx);
#elif TARGET_CPU_X86
  UntypedMDRVA memory(minidump_writer);
  if (!WriteLwpStack(writer_args->lwp_lister,
                     writer_args->sig_ctx->uc_mcontext.gregs[UESP],
                     &memory,
                     &lwp->stack))
    return false;

  TypedMDRVA<MDRawContextX86> context(minidump_writer);
  if (!context.Allocate())
    return false;
  lwp->thread_context = context.location();
  memset(context.get(), 0, sizeof(MDRawContextX86));
  return WriteContext(context.get(),
                      (int*)&writer_args->sig_ctx->uc_mcontext.gregs,
                      &writer_args->sig_ctx->uc_mcontext.fpregs);
#endif
}

bool WriteLwpStream(MinidumpFileWriter* minidump_writer,
                    const SolarisLwp* lwp_lister,
                    const lwpstatus_t* lsp, MDRawThread* lwp) {
  prfpregset_t fp_regs = lsp->pr_fpreg;
  const prgregset_t* gregs = &(lsp->pr_reg);
  UntypedMDRVA memory(minidump_writer);
#if TARGET_CPU_SPARC
  if (!WriteLwpStack(lwp_lister,
                     (*gregs)[R_SP],
                     &memory,
                     &lwp->stack))
    return false;

  // Write context
  TypedMDRVA<MDRawContextSPARC> context(minidump_writer);
  if (!context.Allocate())
    return false;
  // should be the thread_id
  lwp->thread_id = lsp->pr_lwpid;
  lwp->thread_context = context.location();
  memset(context.get(), 0, sizeof(MDRawContextSPARC));
#elif TARGET_CPU_X86
  if (!WriteLwpStack(lwp_lister,
                     (*gregs)[UESP],
                     &memory,
                     &lwp->stack))
  return false;

  // Write context
  TypedMDRVA<MDRawContextX86> context(minidump_writer);
  if (!context.Allocate())
    return false;
  // should be the thread_id
  lwp->thread_id = lsp->pr_lwpid;
  lwp->thread_context = context.location();
  memset(context.get(), 0, sizeof(MDRawContextX86));
#endif /* TARGET_CPU_XXX */
  return WriteContext(context.get(), (int*)gregs, &fp_regs);
}

bool WriteCPUInformation(MDRawSystemInfo* sys_info) {
  struct utsname uts;
  char *major, *minor, *build;

  sys_info->number_of_processors = sysconf(_SC_NPROCESSORS_CONF);
  sys_info->processor_architecture = MD_CPU_ARCHITECTURE_UNKNOWN;
  if (uname(&uts) != -1) {
    // Match "i86pc" as X86 architecture.
    if (strcmp(uts.machine, "i86pc") == 0)
      sys_info->processor_architecture = MD_CPU_ARCHITECTURE_X86;
    else if (strcmp(uts.machine, "sun4u") == 0)
      sys_info->processor_architecture = MD_CPU_ARCHITECTURE_SPARC;
  }

  major = uts.release;
  minor = strchr(major, '.');
  *minor = '\0';
  ++minor;
  sys_info->major_version = atoi(major);
  sys_info->minor_version = atoi(minor);

  build = strchr(uts.version, '_');
  ++build;
  sys_info->build_number = atoi(build);
  
  return true;
}

bool WriteOSInformation(MinidumpFileWriter* minidump_writer,
                        MDRawSystemInfo* sys_info) {
  sys_info->platform_id = MD_OS_SOLARIS;

  struct utsname uts;
  if (uname(&uts) != -1) {
    char os_version[512];
    size_t space_left = sizeof(os_version);
    memset(os_version, 0, space_left);
    const char* os_info_table[] = {
      uts.sysname,
      uts.release,
      uts.version,
      uts.machine,
      "OpenSolaris",
      NULL
    };
    for (const char** cur_os_info = os_info_table;
         *cur_os_info != NULL;
         ++cur_os_info) {
      if (cur_os_info != os_info_table && space_left > 1) {
        strcat(os_version, " ");
        --space_left;
      }
      if (space_left > strlen(*cur_os_info)) {
        strcat(os_version, *cur_os_info);
        space_left -= strlen(*cur_os_info);
      } else {
        break;
      }
    }

    MDLocationDescriptor location;
    if (!minidump_writer->WriteString(os_version, 0, &location))
      return false;
    sys_info->csd_version_rva = location.rva;
  }
  return true;
}

// Callback context for get writting lwp information.
struct LwpInfoCallbackCtx {
  MinidumpFileWriter* minidump_writer;
  const WriterArgument* writer_args;
  TypedMDRVA<MDRawThreadList>* list;
  int lwp_index;
};

bool LwpInformationCallback(lwpstatus_t* lsp, void* context) {
  bool success = true;
  LwpInfoCallbackCtx* callback_context =
    static_cast<LwpInfoCallbackCtx*>(context);

  // The current lwp is the one to handle the crash. Ignore it.
  if (lsp->pr_lwpid != pthread_self()) {
    LwpInfoCallbackCtx* callback_context =
      static_cast<LwpInfoCallbackCtx*>(context);
    MDRawThread lwp;
    memset(&lwp, 0, sizeof(MDRawThread));

    if (lsp->pr_lwpid != callback_context->writer_args->crashed_lwpid ||
        callback_context->writer_args->sig_ctx == NULL) {
      success = WriteLwpStream(callback_context->minidump_writer,
                               callback_context->writer_args->lwp_lister,
                               lsp, &lwp);
    } else {
      success = WriteCrashedLwpStream(callback_context->minidump_writer,
                                      callback_context->writer_args,
                                      lsp, &lwp);
    }
    if (success) {
      callback_context->list->CopyIndexAfterObject(
          callback_context->lwp_index++,
          &lwp, sizeof(MDRawThread));
    }
  }

  return success;
}

bool WriteLwpListStream(MinidumpFileWriter* minidump_writer,
                        const WriterArgument* writer_args,
                        MDRawDirectory* dir) {
  // Get the lwp information.
  const SolarisLwp* lwp_lister = writer_args->lwp_lister;
  int lwp_count = lwp_lister->GetLwpCount();
  if (lwp_count < 0)
    return false;
  TypedMDRVA<MDRawThreadList> list(minidump_writer);
  if (!list.AllocateObjectAndArray(lwp_count - 1, sizeof(MDRawThread)))
    return false;
  dir->stream_type = MD_THREAD_LIST_STREAM;
  dir->location = list.location();
  list.get()->number_of_threads = lwp_count - 1;

  LwpInfoCallbackCtx context;
  context.minidump_writer = minidump_writer;
  context.writer_args = writer_args;
  context.list = &list;
  context.lwp_index = 0;
  CallbackParam<LwpCallback> callback_param(LwpInformationCallback,
                                            &context);
  int written =
    lwp_lister->Lwp_iter_all(lwp_lister->getpid(), &callback_param);
  return written == lwp_count;
}

bool WriteCVRecord(MinidumpFileWriter* minidump_writer,
                   MDRawModule* module,
                   const char* module_path,
                   char* realname) {
  TypedMDRVA<MDCVInfoPDB70> cv(minidump_writer);

  char path[PATH_MAX];
  const char* module_name = module_path ? module_path : "<Unknown>";
  snprintf(path, sizeof(path), "/proc/self/object/%s", module_name);

  size_t module_name_length = strlen(realname);
  if (!cv.AllocateObjectAndArray(module_name_length + 1, sizeof(uint8_t)))
    return false;
  if (!cv.CopyIndexAfterObject(0, realname, module_name_length))
    return false;

  module->cv_record = cv.location();
  MDCVInfoPDB70* cv_ptr = cv.get();
  memset(cv_ptr, 0, sizeof(MDCVInfoPDB70));
  cv_ptr->cv_signature = MD_CVINFOPDB70_SIGNATURE;
  cv_ptr->age = 0;

  // Get the module identifier
  FileID file_id(path);
  unsigned char identifier[16];

  if (file_id.ElfFileIdentifier(identifier)) {
    cv_ptr->signature.data1 = (uint32_t)identifier[0] << 24 |
      (uint32_t)identifier[1] << 16 | (uint32_t)identifier[2] << 8 |
      (uint32_t)identifier[3];
    cv_ptr->signature.data2 = (uint32_t)identifier[4] << 8 | identifier[5];
    cv_ptr->signature.data3 = (uint32_t)identifier[6] << 8 | identifier[7];
    cv_ptr->signature.data4[0] = identifier[8];
    cv_ptr->signature.data4[1] = identifier[9];
    cv_ptr->signature.data4[2] = identifier[10];
    cv_ptr->signature.data4[3] = identifier[11];
    cv_ptr->signature.data4[4] = identifier[12];
    cv_ptr->signature.data4[5] = identifier[13];
    cv_ptr->signature.data4[6] = identifier[14];
    cv_ptr->signature.data4[7] = identifier[15];
  }
  return true;
}

struct ModuleInfoCallbackCtx {
  MinidumpFileWriter* minidump_writer;
  const WriterArgument* writer_args;
  TypedMDRVA<MDRawModuleList>* list;
  int module_index;
};

bool ModuleInfoCallback(const ModuleInfo& module_info, void* context) {
  ModuleInfoCallbackCtx* callback_context =
    static_cast<ModuleInfoCallbackCtx*>(context);
  // Skip those modules without name, or those that are not modules.
  if (strlen(module_info.name) == 0)
    return true;

  MDRawModule module;
  memset(&module, 0, sizeof(module));
  MDLocationDescriptor loc;
  char path[PATH_MAX];
  char buf[PATH_MAX];
  char* realname;
  int count;

  snprintf(path, sizeof (path), "/proc/self/path/%s", module_info.name);
  if ((count = readlink(path, buf, PATH_MAX)) < 0)
    return false;
  buf[count] = '\0';

  if ((realname = strrchr(buf, '/')) == NULL)
    return false;
  realname++;

  if (!callback_context->minidump_writer->WriteString(realname, 0, &loc))
    return false;

  module.base_of_image = (uint64_t)module_info.start_addr;
  module.size_of_image = module_info.size;
  module.module_name_rva = loc.rva;

  if (!WriteCVRecord(callback_context->minidump_writer, &module,
                     module_info.name, realname))
    return false;

  callback_context->list->CopyIndexAfterObject(
      callback_context->module_index++, &module, MD_MODULE_SIZE);
  return true;
}

bool WriteModuleListStream(MinidumpFileWriter* minidump_writer,
                           const WriterArgument* writer_args,
                           MDRawDirectory* dir) {
  TypedMDRVA<MDRawModuleList> list(minidump_writer);
  int module_count = writer_args->lwp_lister->GetModuleCount();

  if (module_count <= 0 ||
      !list.AllocateObjectAndArray(module_count, MD_MODULE_SIZE)) {
    return false;
  }

  dir->stream_type = MD_MODULE_LIST_STREAM;
  dir->location = list.location();
  list.get()->number_of_modules = module_count;
  ModuleInfoCallbackCtx context;
  context.minidump_writer = minidump_writer;
  context.writer_args = writer_args;
  context.list = &list;
  context.module_index = 0;
  CallbackParam<ModuleCallback> callback(ModuleInfoCallback, &context);
  return writer_args->lwp_lister->ListModules(&callback) == module_count;
}

bool WriteSystemInfoStream(MinidumpFileWriter* minidump_writer,
                           const WriterArgument* writer_args,
                           MDRawDirectory* dir) {
  TypedMDRVA<MDRawSystemInfo> sys_info(minidump_writer);

  if (!sys_info.Allocate())
    return false;

  dir->stream_type = MD_SYSTEM_INFO_STREAM;
  dir->location = sys_info.location();

  return WriteCPUInformation(sys_info.get()) &&
         WriteOSInformation(minidump_writer, sys_info.get());
}

bool WriteExceptionStream(MinidumpFileWriter* minidump_writer,
                          const WriterArgument* writer_args,
                          MDRawDirectory* dir) {
  // This happenes when this is not a crash, but a requested dump.
  if (writer_args->sig_ctx == NULL)
    return false;

  TypedMDRVA<MDRawExceptionStream> exception(minidump_writer);
  if (!exception.Allocate())
    return false;

  dir->stream_type = MD_EXCEPTION_STREAM;
  dir->location = exception.location();
  exception.get()->thread_id = writer_args->crashed_lwpid;
  exception.get()->exception_record.exception_code = writer_args->signo;
  exception.get()->exception_record.exception_flags = 0;

#if TARGET_CPU_SPARC
  if (writer_args->sig_ctx != NULL) {
    exception.get()->exception_record.exception_address = 
      writer_args->sig_ctx->uc_mcontext.gregs[REG_PC];
  } else {
    return true;
  }

  // Write context of the exception.
  TypedMDRVA<MDRawContextSPARC> context(minidump_writer);
  if (!context.Allocate())
    return false;
  exception.get()->thread_context = context.location();
  memset(context.get(), 0, sizeof(MDRawContextSPARC));
  return WriteContext(context.get(), writer_args->sig_ctx);
#elif TARGET_CPU_X86
  if (writer_args->sig_ctx != NULL) {
    exception.get()->exception_record.exception_address =
      writer_args->sig_ctx->uc_mcontext.gregs[EIP];
  } else {
    return true;
  }

  // Write context of the exception.
  TypedMDRVA<MDRawContextX86> context(minidump_writer);
  if (!context.Allocate())
    return false;
  exception.get()->thread_context = context.location();
  memset(context.get(), 0, sizeof(MDRawContextX86));
  return WriteContext(context.get(),
                      (int*)&writer_args->sig_ctx->uc_mcontext.gregs,
                      NULL);
#endif
}

bool WriteMiscInfoStream(MinidumpFileWriter* minidump_writer,
                         const WriterArgument* writer_args,
                         MDRawDirectory* dir) {
  TypedMDRVA<MDRawMiscInfo> info(minidump_writer);

  if (!info.Allocate())
    return false;

  dir->stream_type = MD_MISC_INFO_STREAM;
  dir->location = info.location();
  info.get()->size_of_info = sizeof(MDRawMiscInfo);
  info.get()->flags1 = MD_MISCINFO_FLAGS1_PROCESS_ID;
  info.get()->process_id = writer_args->requester_pid;

  return true;
}

bool WriteBreakpadInfoStream(MinidumpFileWriter* minidump_writer,
                             const WriterArgument* writer_args,
                             MDRawDirectory* dir) {
  TypedMDRVA<MDRawBreakpadInfo> info(minidump_writer);

  if (!info.Allocate())
    return false;

  dir->stream_type = MD_BREAKPAD_INFO_STREAM;
  dir->location = info.location();

  info.get()->validity = MD_BREAKPAD_INFO_VALID_DUMP_THREAD_ID |
                         MD_BREAKPAD_INFO_VALID_REQUESTING_THREAD_ID;
  info.get()->dump_thread_id = getpid();
  info.get()->requesting_thread_id = writer_args->requester_pid;
  return true;
}

class AutoLwpResumer {
 public:
  AutoLwpResumer(SolarisLwp* lwp) : lwp_(lwp) {}
  ~AutoLwpResumer() { lwp_->ControlAllLwps(false); }
 private:
  SolarisLwp* lwp_;
};

// Prototype of writer functions.
typedef bool (*WriteStreamFN)(MinidumpFileWriter*,
                              const WriterArgument*,
                              MDRawDirectory*);

// Function table to writer a full minidump.
const WriteStreamFN writers[] = {
  WriteLwpListStream,
  WriteModuleListStream,
  WriteSystemInfoStream,
  WriteExceptionStream,
  WriteMiscInfoStream,
  WriteBreakpadInfoStream,
};

// Will call each writer function in the writers table.
//void* MinidumpGenerator::Write(void* argument) {
void* Write(void* argument) {
  WriterArgument* writer_args = static_cast<WriterArgument*>(argument);

  if (!writer_args->lwp_lister->ControlAllLwps(true))
    return NULL;

  AutoLwpResumer lwpResumer(writer_args->lwp_lister);

  if (writer_args->sighandler_ebp != 0 &&
      writer_args->lwp_lister->FindSigContext(writer_args->sighandler_ebp,
                                              &writer_args->sig_ctx)) {
    writer_args->crashed_stack_bottom = 
      writer_args->lwp_lister->GetLwpStackBottom(
#if TARGET_CPU_SPARC
          writer_args->sig_ctx->uc_mcontext.gregs[REG_O6]
#elif TARGET_CPU_X86
          writer_args->sig_ctx->uc_mcontext.gregs[UESP]
#endif
      );

    int crashed_lwpid = FindCrashingLwp(writer_args->crashed_stack_bottom,
                                        writer_args->requester_pid,
                                        writer_args->lwp_lister);
    if (crashed_lwpid > 0)
      writer_args->crashed_lwpid = crashed_lwpid;
  }

  MinidumpFileWriter* minidump_writer = writer_args->minidump_writer;
  TypedMDRVA<MDRawHeader> header(minidump_writer);
  TypedMDRVA<MDRawDirectory> dir(minidump_writer);
  if (!header.Allocate())
    return 0;

  int writer_count = sizeof(writers) / sizeof(writers[0]);
  // Need directory space for all writers.
  if (!dir.AllocateArray(writer_count))
    return 0;
  header.get()->signature = MD_HEADER_SIGNATURE;
  header.get()->version = MD_HEADER_VERSION;
  header.get()->time_date_stamp = time(NULL);
  header.get()->stream_count = writer_count;
  header.get()->stream_directory_rva = dir.position();

  int dir_index = 0;
  MDRawDirectory local_dir;
  for (int i = 0; i < writer_count; ++i) {
    if ((*writers[i])(minidump_writer, writer_args, &local_dir))
      dir.CopyIndex(dir_index++, &local_dir);
  }

  return 0;
}

}  // namespace

namespace google_breakpad {

MinidumpGenerator::MinidumpGenerator() {
}

MinidumpGenerator::~MinidumpGenerator() {
}

// Write minidump into file.
// It runs in a different thread from the crashing thread.
bool MinidumpGenerator::WriteMinidumpToFile(const char* file_pathname,
                                            int signo,
                                            uintptr_t sighandler_ebp,
                                            ucontext_t** sig_ctx) const {
  // The exception handler thread.
  pthread_t handler_thread;

  assert(file_pathname != NULL);

  if (file_pathname == NULL)
    return false;

  MinidumpFileWriter minidump_writer;
  if (minidump_writer.Open(file_pathname)) {
    WriterArgument argument;
    memset(&argument, 0, sizeof(argument));
    SolarisLwp lwp_lister(getpid());
    argument.lwp_lister = &lwp_lister;
    argument.minidump_writer = &minidump_writer;
    argument.requester_pid = getpid();
    argument.crashed_lwpid = pthread_self();
    argument.signo = signo;
    argument.sighandler_ebp = sighandler_ebp;
    argument.sig_ctx = NULL;

    pthread_create(&handler_thread, NULL, Write, (void*)&argument);
    pthread_join(handler_thread, NULL);
    return true;
  }

  return false;
}

}  // namespace google_breakpad
