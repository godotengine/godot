//===- Win32/Program.cpp - Win32 Program Implementation ------- -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides the Win32 specific implementation of the Program class.
//
//===----------------------------------------------------------------------===//

#include "WindowsSupport.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/ConvertUTF.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/WindowsError.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdio>
#include <fcntl.h>
#include <io.h>
#include <malloc.h>

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only Win32 specific code
//===          and must not be UNIX code
//===----------------------------------------------------------------------===//

namespace llvm {
using namespace sys;

ProcessInfo::ProcessInfo() : ProcessHandle(0), Pid(0), ReturnCode(0) {}

ErrorOr<std::string> sys::findProgramByName(StringRef Name,
                                            ArrayRef<StringRef> Paths) {
#ifdef MSFT_SUPPORTS_CHILD_PROCESSES  // HLSL Change
  assert(!Name.empty() && "Must have a name!");

  if (Name.find_first_of("/\\") != StringRef::npos)
    return std::string(Name);

  const wchar_t *Path = nullptr;
  std::wstring PathStorage;
  if (!Paths.empty()) {
    PathStorage.reserve(Paths.size() * MAX_PATH);
    for (unsigned i = 0; i < Paths.size(); ++i) {
      if (i)
        PathStorage.push_back(L';');
      StringRef P = Paths[i];
      SmallVector<wchar_t, MAX_PATH> TmpPath;
      if (std::error_code EC = windows::UTF8ToUTF16(P, TmpPath))
        return EC;
      PathStorage.append(TmpPath.begin(), TmpPath.end());
    }
    Path = PathStorage.c_str();
  }

  SmallVector<wchar_t, MAX_PATH> U16Name;
  if (std::error_code EC = windows::UTF8ToUTF16(Name, U16Name))
    return EC;

  SmallVector<StringRef, 12> PathExts;
  PathExts.push_back("");
  PathExts.push_back(".exe"); // FIXME: This must be in %PATHEXT%.
  if (const char *PathExtEnv = std::getenv("PATHEXT"))
    SplitString(PathExtEnv, PathExts, ";");

  SmallVector<wchar_t, MAX_PATH> U16Result;
  DWORD Len = MAX_PATH;
  for (StringRef Ext : PathExts) {
    SmallVector<wchar_t, MAX_PATH> U16Ext;
    if (std::error_code EC = windows::UTF8ToUTF16(Ext, U16Ext))
      return EC;

    do {
      U16Result.reserve(Len);
      Len = ::SearchPathW(Path, c_str(U16Name),
                          U16Ext.empty() ? nullptr : c_str(U16Ext),
                          U16Result.capacity(), U16Result.data(), nullptr);
    } while (Len > U16Result.capacity());

    if (Len != 0)
      break; // Found it.
  }

  if (Len == 0)
    return mapWindowsError(::GetLastError());

  U16Result.set_size(Len);

  SmallVector<char, MAX_PATH> U8Result;
  if (std::error_code EC =
          windows::UTF16ToUTF8(U16Result.data(), U16Result.size(), U8Result))
    return EC;

  return std::string(U8Result.begin(), U8Result.end());
#else
  return "";
#endif
}

#ifdef MSFT_SUPPORTS_CHILD_PROCESSES  // HLSL Change
static HANDLE RedirectIO(const StringRef *path, int fd, std::string* ErrMsg) {
  HANDLE h;
  if (path == 0) {
    if (!DuplicateHandle(GetCurrentProcess(), (HANDLE)_get_osfhandle(fd),
                         GetCurrentProcess(), &h,
                         0, TRUE, DUPLICATE_SAME_ACCESS))
      return INVALID_HANDLE_VALUE;
    return h;
  }

  std::string fname;
  if (path->empty())
    fname = "NUL";
  else
    fname = *path;

  SECURITY_ATTRIBUTES sa;
  sa.nLength = sizeof(sa);
  sa.lpSecurityDescriptor = 0;
  sa.bInheritHandle = TRUE;

  SmallVector<wchar_t, 128> fnameUnicode;
  if (path->empty()) {
    // Don't play long-path tricks on "NUL".
    if (windows::UTF8ToUTF16(fname, fnameUnicode))
      return INVALID_HANDLE_VALUE;
  } else {
    if (path::widenPath(fname, fnameUnicode))
      return INVALID_HANDLE_VALUE;
  }
  h = CreateFileW(fnameUnicode.data(), fd ? GENERIC_WRITE : GENERIC_READ,
                  FILE_SHARE_READ, &sa, fd == 0 ? OPEN_EXISTING : CREATE_ALWAYS,
                  FILE_ATTRIBUTE_NORMAL, NULL);
  if (h == INVALID_HANDLE_VALUE) {
    MakeErrMsg(ErrMsg, fname + ": Can't open file for " +
        (fd ? "input: " : "output: "));
  }

  return h;
}

/// ArgNeedsQuotes - Check whether argument needs to be quoted when calling
/// CreateProcess.
static bool ArgNeedsQuotes(const char *Str) {
  return Str[0] == '\0' || strpbrk(Str, "\t \"&\'()*<>\\`^|") != 0;
}

/// CountPrecedingBackslashes - Returns the number of backslashes preceding Cur
/// in the C string Start.
static unsigned int CountPrecedingBackslashes(const char *Start,
                                              const char *Cur) {
  unsigned int Count = 0;
  --Cur;
  while (Cur >= Start && *Cur == '\\') {
    ++Count;
    --Cur;
  }
  return Count;
}

/// EscapePrecedingEscapes - Append a backslash to Dst for every backslash
/// preceding Cur in the Start string.  Assumes Dst has enough space.
static char *EscapePrecedingEscapes(char *Dst, const char *Start,
                                    const char *Cur) {
  unsigned PrecedingEscapes = CountPrecedingBackslashes(Start, Cur);
  while (PrecedingEscapes > 0) {
    *Dst++ = '\\';
    --PrecedingEscapes;
  }
  return Dst;
}

/// ArgLenWithQuotes - Check whether argument needs to be quoted when calling
/// CreateProcess and returns length of quoted arg with escaped quotes
static unsigned int ArgLenWithQuotes(const char *Str) {
  const char *Start = Str;
  bool Quoted = ArgNeedsQuotes(Str);
  unsigned int len = Quoted ? 2 : 0;

  while (*Str != '\0') {
    if (*Str == '\"') {
      // We need to add a backslash, but ensure that it isn't escaped.
      unsigned PrecedingEscapes = CountPrecedingBackslashes(Start, Str);
      len += PrecedingEscapes + 1;
    }
    // Note that we *don't* need to escape runs of backslashes that don't
    // precede a double quote!  See MSDN:
    // http://msdn.microsoft.com/en-us/library/17w5ykft%28v=vs.85%29.aspx

    ++len;
    ++Str;
  }

  if (Quoted) {
    // Make sure the closing quote doesn't get escaped by a trailing backslash.
    unsigned PrecedingEscapes = CountPrecedingBackslashes(Start, Str);
    len += PrecedingEscapes + 1;
  }

  return len;
}

}

static std::unique_ptr<char[]> flattenArgs(const char **args) {
  // First, determine the length of the command line.
  unsigned len = 0;
  for (unsigned i = 0; args[i]; i++) {
    len += ArgLenWithQuotes(args[i]) + 1;
  }

  // Now build the command line.
  std::unique_ptr<char[]> command(new char[len+1]);
  char *p = command.get();

  for (unsigned i = 0; args[i]; i++) {
    const char *arg = args[i];
    const char *start = arg;

    bool needsQuoting = ArgNeedsQuotes(arg);
    if (needsQuoting)
      *p++ = '"';

    while (*arg != '\0') {
      if (*arg == '\"') {
        // Escape all preceding escapes (if any), and then escape the quote.
        p = EscapePrecedingEscapes(p, start, arg);
        *p++ = '\\';
      }

      *p++ = *arg++;
    }

    if (needsQuoting) {
      // Make sure our quote doesn't get escaped by a trailing backslash.
      p = EscapePrecedingEscapes(p, start, arg);
      *p++ = '"';
    }
    *p++ = ' ';
  }

  *p = 0;
  return command;
}

static bool Execute(ProcessInfo &PI, StringRef Program, const char **args,
                    const char **envp, const StringRef **redirects,
                    unsigned memoryLimit, std::string *ErrMsg) {
  if (!sys::fs::can_execute(Program)) {
    if (ErrMsg)
      *ErrMsg = "program not executable";
    return false;
  }

  // Windows wants a command line, not an array of args, to pass to the new
  // process.  We have to concatenate them all, while quoting the args that
  // have embedded spaces (or are empty).
  std::unique_ptr<char[]> command = flattenArgs(args);

  // The pointer to the environment block for the new process.
  std::vector<wchar_t> EnvBlock;

  if (envp) {
    // An environment block consists of a null-terminated block of
    // null-terminated strings. Convert the array of environment variables to
    // an environment block by concatenating them.
    for (unsigned i = 0; envp[i]; ++i) {
      SmallVector<wchar_t, MAX_PATH> EnvString;
      if (std::error_code ec = windows::UTF8ToUTF16(envp[i], EnvString)) {
        SetLastError(ec.value());
        MakeErrMsg(ErrMsg, "Unable to convert environment variable to UTF-16");
        return false;
      }

      EnvBlock.insert(EnvBlock.end(), EnvString.begin(), EnvString.end());
      EnvBlock.push_back(0);
    }
    EnvBlock.push_back(0);
  }

  // Create a child process.
  STARTUPINFOW si;
  memset(&si, 0, sizeof(si));
  si.cb = sizeof(si);
  si.hStdInput = INVALID_HANDLE_VALUE;
  si.hStdOutput = INVALID_HANDLE_VALUE;
  si.hStdError = INVALID_HANDLE_VALUE;

  if (redirects) {
    si.dwFlags = STARTF_USESTDHANDLES;

    si.hStdInput = RedirectIO(redirects[0], 0, ErrMsg);
    if (si.hStdInput == INVALID_HANDLE_VALUE) {
      MakeErrMsg(ErrMsg, "can't redirect stdin");
      return false;
    }
    si.hStdOutput = RedirectIO(redirects[1], 1, ErrMsg);
    if (si.hStdOutput == INVALID_HANDLE_VALUE) {
      CloseHandle(si.hStdInput);
      MakeErrMsg(ErrMsg, "can't redirect stdout");
      return false;
    }
    if (redirects[1] && redirects[2] && *(redirects[1]) == *(redirects[2])) {
      // If stdout and stderr should go to the same place, redirect stderr
      // to the handle already open for stdout.
      if (!DuplicateHandle(GetCurrentProcess(), si.hStdOutput,
                           GetCurrentProcess(), &si.hStdError,
                           0, TRUE, DUPLICATE_SAME_ACCESS)) {
        CloseHandle(si.hStdInput);
        CloseHandle(si.hStdOutput);
        MakeErrMsg(ErrMsg, "can't dup stderr to stdout");
        return false;
      }
    } else {
      // Just redirect stderr
      si.hStdError = RedirectIO(redirects[2], 2, ErrMsg);
      if (si.hStdError == INVALID_HANDLE_VALUE) {
        CloseHandle(si.hStdInput);
        CloseHandle(si.hStdOutput);
        MakeErrMsg(ErrMsg, "can't redirect stderr");
        return false;
      }
    }
  }

  PROCESS_INFORMATION pi;
  memset(&pi, 0, sizeof(pi));

  fflush(stdout);
  fflush(stderr);

  SmallVector<wchar_t, MAX_PATH> ProgramUtf16;
  if (std::error_code ec = path::widenPath(Program, ProgramUtf16)) {
    SetLastError(ec.value());
    MakeErrMsg(ErrMsg,
               std::string("Unable to convert application name to UTF-16"));
    return false;
  }

  SmallVector<wchar_t, MAX_PATH> CommandUtf16;
  if (std::error_code ec = windows::UTF8ToUTF16(command.get(), CommandUtf16)) {
    SetLastError(ec.value());
    MakeErrMsg(ErrMsg,
               std::string("Unable to convert command-line to UTF-16"));
    return false;
  }

  BOOL rc = CreateProcessW(ProgramUtf16.data(), CommandUtf16.data(), 0, 0,
                           TRUE, CREATE_UNICODE_ENVIRONMENT,
                           EnvBlock.empty() ? 0 : EnvBlock.data(), 0, &si,
                           &pi);
  DWORD err = GetLastError();

  // Regardless of whether the process got created or not, we are done with
  // the handles we created for it to inherit.
  CloseHandle(si.hStdInput);
  CloseHandle(si.hStdOutput);
  CloseHandle(si.hStdError);

  // Now return an error if the process didn't get created.
  if (!rc) {
    SetLastError(err);
    MakeErrMsg(ErrMsg, std::string("Couldn't execute program '") +
               Program.str() + "'");
    return false;
  }

  PI.Pid = pi.dwProcessId;
  PI.ProcessHandle = pi.hProcess;

  // Make sure these get closed no matter what.
  ScopedCommonHandle hThread(pi.hThread);

  // Assign the process to a job if a memory limit is defined.
  ScopedJobHandle hJob;
  if (memoryLimit != 0) {
    hJob = CreateJobObjectW(0, 0);
    bool success = false;
    if (hJob) {
      JOBOBJECT_EXTENDED_LIMIT_INFORMATION jeli;
      memset(&jeli, 0, sizeof(jeli));
      jeli.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_PROCESS_MEMORY;
      jeli.ProcessMemoryLimit = uintptr_t(memoryLimit) * 1048576;
      if (SetInformationJobObject(hJob, JobObjectExtendedLimitInformation,
                                  &jeli, sizeof(jeli))) {
        if (AssignProcessToJobObject(hJob, pi.hProcess))
          success = true;
      }
    }
    if (!success) {
      SetLastError(GetLastError());
      MakeErrMsg(ErrMsg, std::string("Unable to set memory limit"));
      TerminateProcess(pi.hProcess, 1);
      WaitForSingleObject(pi.hProcess, INFINITE);
      return false;
    }
  }

  return true;
}

namespace llvm {
ProcessInfo sys::Wait(const ProcessInfo &PI, unsigned SecondsToWait,
                      bool WaitUntilChildTerminates, std::string *ErrMsg) {
  assert(PI.Pid && "invalid pid to wait on, process not started?");
  assert(PI.ProcessHandle &&
         "invalid process handle to wait on, process not started?");
  DWORD milliSecondsToWait = 0;
  if (WaitUntilChildTerminates)
    milliSecondsToWait = INFINITE;
  else if (SecondsToWait > 0)
    milliSecondsToWait = SecondsToWait * 1000;

  ProcessInfo WaitResult = PI;
  DWORD WaitStatus = WaitForSingleObject(PI.ProcessHandle, milliSecondsToWait);
  if (WaitStatus == WAIT_TIMEOUT) {
    if (SecondsToWait) {
      if (!TerminateProcess(PI.ProcessHandle, 1)) {
        if (ErrMsg)
          MakeErrMsg(ErrMsg, "Failed to terminate timed-out program.");

        // -2 indicates a crash or timeout as opposed to failure to execute.
        WaitResult.ReturnCode = -2;
        CloseHandle(PI.ProcessHandle);
        return WaitResult;
      }
      WaitForSingleObject(PI.ProcessHandle, INFINITE);
      CloseHandle(PI.ProcessHandle);
    } else {
      // Non-blocking wait.
      return ProcessInfo();
    }
  }

  // Get its exit status.
  DWORD status;
  BOOL rc = GetExitCodeProcess(PI.ProcessHandle, &status);
  DWORD err = GetLastError();
  if (err != ERROR_INVALID_HANDLE)
    CloseHandle(PI.ProcessHandle);

  if (!rc) {
    SetLastError(err);
    if (ErrMsg)
      MakeErrMsg(ErrMsg, "Failed getting status for program.");

    // -2 indicates a crash or timeout as opposed to failure to execute.
    WaitResult.ReturnCode = -2;
    return WaitResult;
  }

  if (!status)
    return WaitResult;

  // Pass 10(Warning) and 11(Error) to the callee as negative value.
  if ((status & 0xBFFF0000U) == 0x80000000U)
    WaitResult.ReturnCode = static_cast<int>(status);
  else if (status & 0xFF)
    WaitResult.ReturnCode = status & 0x7FFFFFFF;
  else
    WaitResult.ReturnCode = 1;

  return WaitResult;
}

#endif // HLSL Change - MSFT_SUPPORTS_CHILD_PROCESSES

std::error_code sys::ChangeStdinToBinary() {
  int result = _setmode(_fileno(stdin), _O_BINARY);
  if (result == -1)
    return std::error_code(errno, std::generic_category());
  return std::error_code();
}

std::error_code sys::ChangeStdoutToBinary() {
  int result = _setmode(_fileno(stdout), _O_BINARY);
  if (result == -1)
    return std::error_code(errno, std::generic_category());
  return std::error_code();
}

std::error_code
llvm::sys::writeFileWithEncoding(StringRef FileName, StringRef Contents,
                                 WindowsEncodingMethod Encoding) {
  std::error_code EC;
  llvm::raw_fd_ostream OS(FileName, EC, llvm::sys::fs::OpenFlags::F_Text);
  if (EC)
    return EC;

  if (Encoding == WEM_UTF8) {
    OS << Contents;
  } else if (Encoding == WEM_CurrentCodePage) {
    SmallVector<wchar_t, 1> ArgsUTF16;
    SmallVector<char, 1> ArgsCurCP;

    if ((EC = windows::UTF8ToUTF16(Contents, ArgsUTF16)))
      return EC;

    if ((EC = windows::UTF16ToCurCP(
             ArgsUTF16.data(), ArgsUTF16.size(), ArgsCurCP)))
      return EC;

    OS.write(ArgsCurCP.data(), ArgsCurCP.size());
  } else if (Encoding == WEM_UTF16) {
    SmallVector<wchar_t, 1> ArgsUTF16;

    if ((EC = windows::UTF8ToUTF16(Contents, ArgsUTF16)))
      return EC;

    // Endianness guessing
    char BOM[2];
    uint16_t src = UNI_UTF16_BYTE_ORDER_MARK_NATIVE;
    memcpy(BOM, &src, 2);
    OS.write(BOM, 2);
    OS.write((char *)ArgsUTF16.data(), ArgsUTF16.size() << 1);
  } else {
    llvm_unreachable("Unknown encoding");
  }

  if (OS.has_error())
    return make_error_code(errc::io_error);

  return EC;
}

#ifdef MSFT_SUPPORTS_CHILD_PROCESSES  // HLSL Change

bool llvm::sys::argumentsFitWithinSystemLimits(ArrayRef<const char*> Args) {
  // The documented max length of the command line passed to CreateProcess.
  static const size_t MaxCommandStringLength = 32768;
  size_t ArgLength = 0;
  for (ArrayRef<const char*>::iterator I = Args.begin(), E = Args.end();
       I != E; ++I) {
    // Account for the trailing space for every arg but the last one and the
    // trailing NULL of the last argument.
    ArgLength += ArgLenWithQuotes(*I) + 1;
    if (ArgLength > MaxCommandStringLength) {
      return false;
    }
  }
  return true;
}

#endif // MSFT_SUPPORTS_CHILD_PROCESSES  // HLSL Change

}
