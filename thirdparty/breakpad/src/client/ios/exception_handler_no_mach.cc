// Copyright 2006 Google LLC
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

#ifdef HAVE_CONFIG_H
#include <config.h>  // Must come first
#endif

#include <signal.h>
#include <TargetConditionals.h>

#include "client/mac/handler/minidump_generator.h"
#include "client/ios/exception_handler_no_mach.h"

#ifndef USE_PROTECTED_ALLOCATIONS
#if TARGET_OS_TV
#define USE_PROTECTED_ALLOCATIONS 1
#else
#define USE_PROTECTED_ALLOCATIONS 0
#endif
#endif

// If USE_PROTECTED_ALLOCATIONS is activated then the
// gBreakpadAllocator needs to be setup in other code
// ahead of time.  Please see ProtectedMemoryAllocator.h
// for more details.
#if USE_PROTECTED_ALLOCATIONS
  #include "client/mac/handler/protected_memory_allocator.h"
  extern ProtectedMemoryAllocator* gBreakpadAllocator;
#endif

namespace google_breakpad {

const int kExceptionSignals[] = {
   // Core-generating signals.
  SIGABRT, SIGBUS, SIGFPE, SIGILL, SIGQUIT, SIGSEGV, SIGSYS, SIGTRAP, SIGEMT,
  SIGXCPU, SIGXFSZ,
  // Non-core-generating but terminating signals.
  SIGALRM, SIGHUP, SIGINT, SIGPIPE, SIGPROF, SIGTERM, SIGUSR1, SIGUSR2,
  SIGVTALRM, SIGXCPU, SIGXFSZ, SIGIO,
};
const int kNumHandledSignals =
    sizeof(kExceptionSignals) / sizeof(kExceptionSignals[0]);
struct scoped_ptr<struct sigaction> old_handlers[kNumHandledSignals];

static union {
#if USE_PROTECTED_ALLOCATIONS
#if defined PAGE_MAX_SIZE
  char protected_buffer[PAGE_MAX_SIZE] __attribute__((aligned(PAGE_MAX_SIZE)));
#else
  char protected_buffer[PAGE_SIZE] __attribute__((aligned(PAGE_SIZE)));
#endif  // defined PAGE_MAX_SIZE
#endif  // USE_PROTECTED_ALLOCATIONS
  google_breakpad::ExceptionHandler* handler;
} gProtectedData;

ExceptionHandler::ExceptionHandler(const string& dump_path,
                                   FilterCallback filter,
                                   MinidumpCallback callback,
                                   void* callback_context,
                                   bool install_handler,
                                   const char* port_name)
    : dump_path_(),
      filter_(filter),
      callback_(callback),
      callback_context_(callback_context),
      directCallback_(NULL),
      installed_exception_handler_(false),
      is_in_teardown_(false) {
  // This will update to the ID and C-string pointers
  set_dump_path(dump_path);
  MinidumpGenerator::GatherSystemInformation();
  Setup();
}

// special constructor if we want to bypass minidump writing and
// simply get a callback with the exception information
ExceptionHandler::ExceptionHandler(DirectCallback callback,
                                   void* callback_context,
                                   bool install_handler)
    : dump_path_(),
      filter_(NULL),
      callback_(NULL),
      callback_context_(callback_context),
      directCallback_(callback),
      installed_exception_handler_(false),
      is_in_teardown_(false) {
  MinidumpGenerator::GatherSystemInformation();
  Setup();
}

ExceptionHandler::~ExceptionHandler() {
  Teardown();
}

bool ExceptionHandler::WriteMinidumpWithException(
    int exception_type,
    int exception_code,
    int exception_subcode,
    breakpad_ucontext_t* task_context,
    mach_port_t thread_name,
    bool exit_after_write,
    bool report_current_thread) {
  bool result = false;

#if !TARGET_OS_TV
  exit_after_write = false;
#endif  // !TARGET_OS_TV

  if (directCallback_) {
    if (directCallback_(callback_context_,
                        exception_type,
                        exception_code,
                        exception_subcode,
                        thread_name) ) {
      if (exit_after_write)
        _exit(exception_type);
    }
  } else {
    string minidump_id;

    // Putting the MinidumpGenerator in its own context will ensure that the
    // destructor is executed, closing the newly created minidump file.
    if (!dump_path_.empty()) {
      MinidumpGenerator md(mach_task_self(),
                           report_current_thread ? MACH_PORT_NULL :
                                                   mach_thread_self());
      md.SetTaskContext(task_context);
      if (exception_type && exception_code) {
        // If this is a real exception, give the filter (if any) a chance to
        // decide if this should be sent.
        if (filter_ && !filter_(callback_context_))
          return false;

        md.SetExceptionInformation(exception_type, exception_code,
                                   exception_subcode, thread_name);
      }

      result = md.Write(next_minidump_path_c_);
    }

    // Call user specified callback (if any)
    if (callback_) {
      // If the user callback returned true and we're handling an exception
      // (rather than just writing out the file), then we should exit without
      // forwarding the exception to the next handler.
      if (callback_(dump_path_c_, next_minidump_id_c_, callback_context_,
                    result)) {
        if (exit_after_write)
          _exit(exception_type);
      }
    }
  }

  return result;
}

// static
void ExceptionHandler::SignalHandler(int sig, siginfo_t* info, void* uc) {
#if USE_PROTECTED_ALLOCATIONS
  if (gBreakpadAllocator)
    gBreakpadAllocator->Unprotect();
#endif
  gProtectedData.handler->WriteMinidumpWithException(
      EXC_SOFTWARE,
      MD_EXCEPTION_CODE_MAC_ABORT,
      0,
      static_cast<breakpad_ucontext_t*>(uc),
      mach_thread_self(),
      true,
      true);
#if USE_PROTECTED_ALLOCATIONS
  if (gBreakpadAllocator)
    gBreakpadAllocator->Protect();
#endif
}

bool ExceptionHandler::InstallHandlers() {
  // If a handler is already installed, something is really wrong.
  if (gProtectedData.handler != NULL)
    return false;
  for (int i = 0; i < kNumHandledSignals; ++i) {
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sigemptyset(&sa.sa_mask);
    sigaddset(&sa.sa_mask, kExceptionSignals[i]);
    sa.sa_sigaction = ExceptionHandler::SignalHandler;
    sa.sa_flags = SA_ONSTACK | SA_SIGINFO;

    if (sigaction(kExceptionSignals[i], &sa, old_handlers[i].get()) == -1) {
      return false;
    }
  }
  gProtectedData.handler = this;
#if USE_PROTECTED_ALLOCATIONS
  assert(((size_t)(gProtectedData.protected_buffer) & PAGE_MASK) == 0);
  mprotect(gProtectedData.protected_buffer, PAGE_SIZE, PROT_READ);
#endif  // USE_PROTECTED_ALLOCATIONS
  installed_exception_handler_ = true;
  return true;
}

bool ExceptionHandler::UninstallHandlers() {
  for (int i = 0; i < kNumHandledSignals; ++i) {
    if (old_handlers[i].get()) {
      sigaction(kExceptionSignals[i], old_handlers[i].get(), NULL);
      old_handlers[i].reset();
    }
  }
#if USE_PROTECTED_ALLOCATIONS
  mprotect(gProtectedData.protected_buffer, PAGE_SIZE, PROT_READ | PROT_WRITE);
#endif  // USE_PROTECTED_ALLOCATIONS
  gProtectedData.handler = NULL;
  installed_exception_handler_ = false;
  return true;
}

bool ExceptionHandler::Setup() {
  if (!InstallHandlers())
    return false;
  return true;
}

bool ExceptionHandler::Teardown() {
  is_in_teardown_ = true;

  if (!UninstallHandlers())
    return false;

  return true;
}

void ExceptionHandler::UpdateNextID() {
  next_minidump_path_ =
    (MinidumpGenerator::UniqueNameInDirectory(dump_path_, &next_minidump_id_));

  next_minidump_path_c_ = next_minidump_path_.c_str();
  next_minidump_id_c_ = next_minidump_id_.c_str();
}

}  // namespace google_breakpad
