//===-- Program.cpp - Implement OS Program Concept --------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements the operating system Program concept.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Program.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Config/config.h"
#include <system_error>
using namespace llvm;
using namespace sys;

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only TRULY operating system
//===          independent code.
//===----------------------------------------------------------------------===//

#if defined(MSFT_SUPPORTS_CHILD_PROCESSES) || defined(LLVM_ON_UNIX)

static bool Execute(ProcessInfo &PI, StringRef Program, const char **args,
                    const char **env, const StringRef **Redirects,
                    unsigned memoryLimit, std::string *ErrMsg);

int sys::ExecuteAndWait(StringRef Program, const char **args, const char **envp,
                        const StringRef **redirects, unsigned secondsToWait,
                        unsigned memoryLimit, std::string *ErrMsg,
                        bool *ExecutionFailed) {
  ProcessInfo PI;
  if (Execute(PI, Program, args, envp, redirects, memoryLimit, ErrMsg)) {
    if (ExecutionFailed)
      *ExecutionFailed = false;
    ProcessInfo Result = Wait(
        PI, secondsToWait, /*WaitUntilTerminates=*/secondsToWait == 0, ErrMsg);
    return Result.ReturnCode;
  }

  if (ExecutionFailed)
    *ExecutionFailed = true;

  return -1;
}

ProcessInfo sys::ExecuteNoWait(StringRef Program, const char **args,
                               const char **envp, const StringRef **redirects,
                               unsigned memoryLimit, std::string *ErrMsg,
                               bool *ExecutionFailed) {
  ProcessInfo PI;
  if (ExecutionFailed)
    *ExecutionFailed = false;
  if (!Execute(PI, Program, args, envp, redirects, memoryLimit, ErrMsg))
    if (ExecutionFailed)
      *ExecutionFailed = true;

  return PI;
}

#endif // MSFT_SUPPORTS_CHILD_PROCESSES

// Include the platform-specific parts of this class.
#ifdef LLVM_ON_UNIX
#include "Unix/Program.inc"
#endif
#ifdef LLVM_ON_WIN32
#include "Windows/Program.inc"
#endif
