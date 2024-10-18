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

#ifndef CLIENT_SOLARIS_HANDLER_SOLARIS_LWP_H__
#define CLIENT_SOLARIS_HANDLER_SOLARIS_LWP_H__

#if defined(sparc) || defined(__sparc)
#define TARGET_CPU_SPARC 1
#elif defined(i386) || defined(__i386)
#define TARGET_CPU_X86 1
#else
#error "cannot determine cpu type"
#endif

#include <signal.h>
#include <stdint.h>
#include <sys/user.h>
#include <ucontext.h>

#ifndef _KERNEL
#define _KERNEL
#define MUST_UNDEF_KERNEL
#endif  // _KERNEL
#include <sys/procfs.h>
#ifdef MUST_UNDEF_KERNEL
#undef _KERNEL
#undef MUST_UNDEF_KERNEL
#endif  // MUST_UNDEF_KERNEL

namespace google_breakpad {

// Max module path name length.
static const int kMaxModuleNameLength = 256;

// Holding infomaton about a module in the process.
struct ModuleInfo {
  char name[kMaxModuleNameLength];
  uintptr_t start_addr;
  int size;
};

// A callback to run when getting a lwp in the process.
// Return true will go on to the next lwp while return false will stop the
// iteration.
typedef bool (*LwpCallback)(lwpstatus_t* lsp, void* context);

// A callback to run when a new module is found in the process.
// Return true will go on to the next module while return false will stop the
// iteration.
typedef bool (*ModuleCallback)(const ModuleInfo& module_info, void* context);

// A callback to run when getting a lwpid in the process.
// Return true will go on to the next lwp while return false will stop the
// iteration.
typedef bool (*LwpidCallback)(int lwpid, void* context);

// Holding the callback information.
template<class CallbackFunc>
struct CallbackParam {
  // Callback function address.
  CallbackFunc call_back;
  // Callback context;
  void* context;

  CallbackParam() : call_back(NULL), context(NULL) {
  }

  CallbackParam(CallbackFunc func, void* func_context) :
    call_back(func), context(func_context) {
  }
};

///////////////////////////////////////////////////////////////////////////////

//
// SolarisLwp
//
// Provides handy support for operation on Solaris lwps.
// It uses proc file system to get lwp information.
//
// TODO(Alfred): Currently it only supports x86. Add SPARC support.
//
class SolarisLwp {
 public:
  // Create a SolarisLwp instance to list all the lwps in a process.
  explicit SolarisLwp(int pid);
  ~SolarisLwp();

  int getpid() const { return this->pid_; }

  // Control all the lwps in the process.
  // Return the number of suspended/resumed lwps in the process.
  // Return -1 means failed to control lwps.
  int ControlAllLwps(bool suspend);

  // Get the count of lwps in the process.
  // Return -1 means error.
  int GetLwpCount() const;

  // Iterate the lwps of process.
  // Whenever there is a lwp found, the callback will be invoked to process
  // the information.
  // Return the callback return value or -1 on error.
  int Lwp_iter_all(int pid, CallbackParam<LwpCallback>* callback_param) const;

  // Get the module count of the current process.
  int GetModuleCount() const;

  // Get the mapped modules in the address space.
  // Whenever a module is found, the callback will be invoked to process the
  // information.
  // Return how may modules are found.
  int ListModules(CallbackParam<ModuleCallback>* callback_param) const;

  // Get the bottom of the stack from esp.
  uintptr_t GetLwpStackBottom(uintptr_t current_esp) const;

  // Finds a signal context on the stack given the ebp of our signal handler.
  bool FindSigContext(uintptr_t sighandler_ebp, ucontext_t** sig_ctx);

 private:
  // Check if the address is a valid virtual address.
  bool IsAddressMapped(uintptr_t address) const;

 private:
  // The pid of the process we are listing lwps.
  int pid_;
};

}  // namespace google_breakpad

#endif  // CLIENT_SOLARIS_HANDLER_SOLARIS_LWP_H__
