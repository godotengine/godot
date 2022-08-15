//===- Win32/Signals.cpp - Win32 Signals Implementation ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides the Win32 specific implementation of the Signals class.
//
//===----------------------------------------------------------------------===//

#if 0 // HLSL Change - disable signal processing

#include "llvm/Support/FileSystem.h"
#include <algorithm>
#include <signal.h>
#include <stdio.h>
#include <vector>

#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

// The Windows.h header must be after LLVM and standard headers.
#include "WindowsSupport.h"

#ifdef __MINGW32__
 #include <imagehlp.h>
#else
 #include <dbghelp.h>
#endif
#include <psapi.h>

#ifdef _MSC_VER
 #pragma comment(lib, "psapi.lib")
#elif __MINGW32__
 #if (HAVE_LIBPSAPI != 1)
  #error "libpsapi.a should be present"
 #endif
 // The version of g++ that comes with MinGW does *not* properly understand
 // the ll format specifier for printf. However, MinGW passes the format
 // specifiers on to the MSVCRT entirely, and the CRT understands the ll
 // specifier. So these warnings are spurious in this case. Since we compile
 // with -Wall, this will generate these warnings which should be ignored. So
 // we will turn off the warnings for this just file. However, MinGW also does
 // not support push and pop for diagnostics, so we have to manually turn it
 // back on at the end of the file.
 #pragma GCC diagnostic ignored "-Wformat"
 #pragma GCC diagnostic ignored "-Wformat-extra-args"

 #if !defined(__MINGW64_VERSION_MAJOR)
 // MinGW.org does not have updated support for the 64-bit versions of the
 // DebugHlp APIs. So we will have to load them manually. The structures and
 // method signatures were pulled from DbgHelp.h in the Windows Platform SDK,
 // and adjusted for brevity.
 typedef struct _IMAGEHLP_LINE64 {
   DWORD    SizeOfStruct;
   PVOID    Key;
   DWORD    LineNumber;
   PCHAR    FileName;
   DWORD64  Address;
 } IMAGEHLP_LINE64, *PIMAGEHLP_LINE64;

 typedef struct _IMAGEHLP_SYMBOL64 {
   DWORD   SizeOfStruct;
   DWORD64 Address;
   DWORD   Size;
   DWORD   Flags;
   DWORD   MaxNameLength;
   CHAR    Name[1];
 } IMAGEHLP_SYMBOL64, *PIMAGEHLP_SYMBOL64;

 typedef struct _tagADDRESS64 {
   DWORD64       Offset;
   WORD          Segment;
   ADDRESS_MODE  Mode;
 } ADDRESS64, *LPADDRESS64;

 typedef struct _KDHELP64 {
   DWORD64   Thread;
   DWORD   ThCallbackStack;
   DWORD   ThCallbackBStore;
   DWORD   NextCallback;
   DWORD   FramePointer;
   DWORD64   KiCallUserMode;
   DWORD64   KeUserCallbackDispatcher;
   DWORD64   SystemRangeStart;
   DWORD64   KiUserExceptionDispatcher;
   DWORD64   StackBase;
   DWORD64   StackLimit;
   DWORD64   Reserved[5];
 } KDHELP64, *PKDHELP64;

 typedef struct _tagSTACKFRAME64 {
   ADDRESS64   AddrPC;
   ADDRESS64   AddrReturn;
   ADDRESS64   AddrFrame;
   ADDRESS64   AddrStack;
   ADDRESS64   AddrBStore;
   PVOID       FuncTableEntry;
   DWORD64     Params[4];
   BOOL        Far;
   BOOL        Virtual;
   DWORD64     Reserved[3];
   KDHELP64    KdHelp;
 } STACKFRAME64, *LPSTACKFRAME64;
 #endif // !defined(__MINGW64_VERSION_MAJOR)
#endif // __MINGW32__

typedef BOOL (__stdcall *PREAD_PROCESS_MEMORY_ROUTINE64)(HANDLE hProcess,
                      DWORD64 qwBaseAddress, PVOID lpBuffer, DWORD nSize,
                      LPDWORD lpNumberOfBytesRead);

typedef PVOID (__stdcall *PFUNCTION_TABLE_ACCESS_ROUTINE64)( HANDLE ahProcess,
                      DWORD64 AddrBase);

typedef DWORD64 (__stdcall *PGET_MODULE_BASE_ROUTINE64)(HANDLE hProcess,
                      DWORD64 Address);

typedef DWORD64 (__stdcall *PTRANSLATE_ADDRESS_ROUTINE64)(HANDLE hProcess,
                      HANDLE hThread, LPADDRESS64 lpaddr);

typedef BOOL (WINAPI *fpStackWalk64)(DWORD, HANDLE, HANDLE, LPSTACKFRAME64,
                      PVOID, PREAD_PROCESS_MEMORY_ROUTINE64,
                      PFUNCTION_TABLE_ACCESS_ROUTINE64,
                      PGET_MODULE_BASE_ROUTINE64,
                      PTRANSLATE_ADDRESS_ROUTINE64);
static fpStackWalk64 fStackWalk64;

typedef DWORD64 (WINAPI *fpSymGetModuleBase64)(HANDLE, DWORD64);
static fpSymGetModuleBase64 fSymGetModuleBase64;

typedef BOOL (WINAPI *fpSymGetSymFromAddr64)(HANDLE, DWORD64,
                      PDWORD64, PIMAGEHLP_SYMBOL64);
static fpSymGetSymFromAddr64 fSymGetSymFromAddr64;

typedef BOOL (WINAPI *fpSymGetLineFromAddr64)(HANDLE, DWORD64,
                      PDWORD, PIMAGEHLP_LINE64);
static fpSymGetLineFromAddr64 fSymGetLineFromAddr64;

typedef PVOID (WINAPI *fpSymFunctionTableAccess64)(HANDLE, DWORD64);
static fpSymFunctionTableAccess64 fSymFunctionTableAccess64;

typedef DWORD (WINAPI *fpSymSetOptions)(DWORD);
static fpSymSetOptions fSymSetOptions;

typedef BOOL (WINAPI *fpSymInitialize)(HANDLE, PCSTR, BOOL);
static fpSymInitialize fSymInitialize;

static bool load64BitDebugHelp(void) {
  HMODULE hLib = ::LoadLibraryW(L"Dbghelp.dll");
  if (hLib) {
    fStackWalk64 = (fpStackWalk64)
                      ::GetProcAddress(hLib, "StackWalk64");
    fSymGetModuleBase64 = (fpSymGetModuleBase64)
                      ::GetProcAddress(hLib, "SymGetModuleBase64");
    fSymGetSymFromAddr64 = (fpSymGetSymFromAddr64)
                      ::GetProcAddress(hLib, "SymGetSymFromAddr64");
    fSymGetLineFromAddr64 = (fpSymGetLineFromAddr64)
                      ::GetProcAddress(hLib, "SymGetLineFromAddr64");
    fSymFunctionTableAccess64 = (fpSymFunctionTableAccess64)
                     ::GetProcAddress(hLib, "SymFunctionTableAccess64");
    fSymSetOptions = (fpSymSetOptions)::GetProcAddress(hLib, "SymSetOptions");
    fSymInitialize = (fpSymInitialize)::GetProcAddress(hLib, "SymInitialize");
  }
  return fStackWalk64 && fSymInitialize && fSymSetOptions;
}

// Forward declare.
static LONG WINAPI LLVMUnhandledExceptionFilter(LPEXCEPTION_POINTERS ep);
static BOOL WINAPI LLVMConsoleCtrlHandler(DWORD dwCtrlType);

// InterruptFunction - The function to call if ctrl-c is pressed.
static void (*InterruptFunction)() = 0;

static std::vector<std::string> *FilesToRemove = NULL;
static std::vector<std::pair<void(*)(void*), void*> > *CallBacksToRun = 0;
static bool RegisteredUnhandledExceptionFilter = false;
static bool CleanupExecuted = false;
static PTOP_LEVEL_EXCEPTION_FILTER OldFilter = NULL;

// Windows creates a new thread to execute the console handler when an event
// (such as CTRL/C) occurs.  This causes concurrency issues with the above
// globals which this critical section addresses.
static CRITICAL_SECTION CriticalSection;
static bool CriticalSectionInitialized = false;

static void PrintStackTraceForThread(llvm::raw_ostream &OS, HANDLE hProcess,
                                     HANDLE hThread, STACKFRAME64 &StackFrame,
                                     CONTEXT *Context) {
  DWORD machineType;
#if defined(_M_X64)
  machineType = IMAGE_FILE_MACHINE_AMD64;
#else
  machineType = IMAGE_FILE_MACHINE_I386;
#endif

  // Initialize the symbol handler.
  fSymSetOptions(SYMOPT_DEFERRED_LOADS | SYMOPT_LOAD_LINES);
  fSymInitialize(hProcess, NULL, TRUE);

  while (true) {
    if (!fStackWalk64(machineType, hProcess, hThread, &StackFrame, Context, 0,
                      fSymFunctionTableAccess64, fSymGetModuleBase64, 0)) {
      break;
    }

    if (StackFrame.AddrFrame.Offset == 0)
      break;

    using namespace llvm;
    // Print the PC in hexadecimal.
    DWORD64 PC = StackFrame.AddrPC.Offset;
#if defined(_M_X64)
    OS << format("0x%016llX", PC);
#elif defined(_M_IX86)
    OS << format("0x%08lX", static_cast<DWORD>(PC));
#endif

// Print the parameters.  Assume there are four.
#if defined(_M_X64)
    OS << format(" (0x%016llX 0x%016llX 0x%016llX 0x%016llX)",
            StackFrame.Params[0], StackFrame.Params[1], StackFrame.Params[2],
            StackFrame.Params[3]);
#elif defined(_M_IX86)
    OS << format(" (0x%08lX 0x%08lX 0x%08lX 0x%08lX)",
            static_cast<DWORD>(StackFrame.Params[0]),
            static_cast<DWORD>(StackFrame.Params[1]),
            static_cast<DWORD>(StackFrame.Params[2]),
            static_cast<DWORD>(StackFrame.Params[3]));
#endif
    // Verify the PC belongs to a module in this process.
    if (!fSymGetModuleBase64(hProcess, PC)) {
      OS << " <unknown module>\n";
      continue;
    }

    // Print the symbol name.
    char buffer[512];
    IMAGEHLP_SYMBOL64 *symbol = reinterpret_cast<IMAGEHLP_SYMBOL64 *>(buffer);
    memset(symbol, 0, sizeof(IMAGEHLP_SYMBOL64));
    symbol->SizeOfStruct = sizeof(IMAGEHLP_SYMBOL64);
    symbol->MaxNameLength = 512 - sizeof(IMAGEHLP_SYMBOL64);

    DWORD64 dwDisp;
    if (!fSymGetSymFromAddr64(hProcess, PC, &dwDisp, symbol)) {
      OS << '\n';
      continue;
    }

    buffer[511] = 0;
    if (dwDisp > 0)
      OS << format(", %s() + 0x%llX bytes(s)", (const char*)symbol->Name,
                   dwDisp);
    else
      OS << format(", %s", (const char*)symbol->Name);

    // Print the source file and line number information.
    IMAGEHLP_LINE64 line = {};
    DWORD dwLineDisp;
    line.SizeOfStruct = sizeof(line);
    if (fSymGetLineFromAddr64(hProcess, PC, &dwLineDisp, &line)) {
      OS << format(", %s, line %lu", line.FileName, line.LineNumber);
      if (dwLineDisp > 0)
        OS << format(" + 0x%lX byte(s)", dwLineDisp);
    }

    OS << '\n';
  }
}

namespace llvm {

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only Win32 specific code
//===          and must not be UNIX code
//===----------------------------------------------------------------------===//

#ifdef _MSC_VER
/// AvoidMessageBoxHook - Emulates hitting "retry" from an "abort, retry,
/// ignore" CRT debug report dialog.  "retry" raises an exception which
/// ultimately triggers our stack dumper.
static LLVM_ATTRIBUTE_UNUSED int
__cdecl // HLSL Change
AvoidMessageBoxHook(int ReportType, char *Message, int *Return) {
  // Set *Return to the retry code for the return value of _CrtDbgReport:
  // http://msdn.microsoft.com/en-us/library/8hyw4sy7(v=vs.71).aspx
  // This may also trigger just-in-time debugging via DebugBreak().
  if (Return)
    *Return = 1;
  // Don't call _CrtDbgReport.
  return TRUE;
}

#endif

extern "C" void HandleAbort(int Sig) {
  if (Sig == SIGABRT) {
    LLVM_BUILTIN_TRAP;
  }
}

static void InitializeThreading() {
  if (CriticalSectionInitialized)
    return;

  // Now's the time to create the critical section. This is the first time
  // through here, and there's only one thread.
  InitializeCriticalSection(&CriticalSection);
  CriticalSectionInitialized = true;
}

static void RegisterHandler() {
  // If we cannot load up the APIs (which would be unexpected as they should
  // exist on every version of Windows we support), we will bail out since
  // there would be nothing to report.
  if (!load64BitDebugHelp()) {
    assert(false && "These APIs should always be available");
    return;
  }

  if (RegisteredUnhandledExceptionFilter) {
    EnterCriticalSection(&CriticalSection);
    return;
  }

  InitializeThreading();

  // Enter it immediately.  Now if someone hits CTRL/C, the console handler
  // can't proceed until the globals are updated.
  EnterCriticalSection(&CriticalSection);

  RegisteredUnhandledExceptionFilter = true;
  OldFilter = SetUnhandledExceptionFilter(LLVMUnhandledExceptionFilter);
  SetConsoleCtrlHandler(LLVMConsoleCtrlHandler, TRUE);

  // IMPORTANT NOTE: Caller must call LeaveCriticalSection(&CriticalSection) or
  // else multi-threading problems will ensue.
}

// RemoveFileOnSignal - The public API
bool sys::RemoveFileOnSignal(StringRef Filename, std::string* ErrMsg) {
  RegisterHandler();

  if (CleanupExecuted) {
    if (ErrMsg)
      *ErrMsg = "Process terminating -- cannot register for removal";
    return true;
  }

  if (FilesToRemove == NULL)
    FilesToRemove = new std::vector<std::string>;

  FilesToRemove->push_back(Filename);

  LeaveCriticalSection(&CriticalSection);
  return false;
}

// DontRemoveFileOnSignal - The public API
void sys::DontRemoveFileOnSignal(StringRef Filename) {
  if (FilesToRemove == NULL)
    return;

  RegisterHandler();

  std::vector<std::string>::reverse_iterator I =
  std::find(FilesToRemove->rbegin(), FilesToRemove->rend(), Filename);
  if (I != FilesToRemove->rend())
    FilesToRemove->erase(I.base()-1);

  LeaveCriticalSection(&CriticalSection);
}

void sys::DisableSystemDialogsOnCrash() {
  // Crash to stack trace handler on abort.
  signal(SIGABRT, HandleAbort);

  // The following functions are not reliably accessible on MinGW.
#ifdef _MSC_VER
  // We're already handling writing a "something went wrong" message.
  _set_abort_behavior(0, _WRITE_ABORT_MSG);
  // Disable Dr. Watson.
  _set_abort_behavior(0, _CALL_REPORTFAULT);
  _CrtSetReportHook(AvoidMessageBoxHook);
#endif

  // Disable standard error dialog box.
  SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX |
               SEM_NOOPENFILEERRORBOX);
  _set_error_mode(_OUT_TO_STDERR);
}

/// PrintStackTraceOnErrorSignal - When an error signal (such as SIBABRT or
/// SIGSEGV) is delivered to the process, print a stack trace and then exit.
void sys::PrintStackTraceOnErrorSignal(bool DisableCrashReporting) {
  DisableSystemDialogsOnCrash();
  RegisterHandler();
  LeaveCriticalSection(&CriticalSection);
}
}

#if defined(__MINGW32__) && !defined(__MINGW64_VERSION_MAJOR)
// Provide a prototype for RtlCaptureContext, mingw32 from mingw.org is
// missing it but mingw-w64 has it.
extern "C" VOID WINAPI RtlCaptureContext(PCONTEXT ContextRecord);
#endif

void llvm::sys::PrintStackTrace(raw_ostream &OS) {

  STACKFRAME64 StackFrame = {};
  CONTEXT Context = {};
  ::RtlCaptureContext(&Context);
#if defined(_M_X64)
  StackFrame.AddrPC.Offset = Context.Rip;
  StackFrame.AddrStack.Offset = Context.Rsp;
  StackFrame.AddrFrame.Offset = Context.Rbp;
#else
  StackFrame.AddrPC.Offset = Context.Eip;
  StackFrame.AddrStack.Offset = Context.Esp;
  StackFrame.AddrFrame.Offset = Context.Ebp;
#endif
  StackFrame.AddrPC.Mode = AddrModeFlat;
  StackFrame.AddrStack.Mode = AddrModeFlat;
  StackFrame.AddrFrame.Mode = AddrModeFlat;
  PrintStackTraceForThread(OS, GetCurrentProcess(), GetCurrentThread(),
                           StackFrame, &Context);
}


void llvm::sys::SetInterruptFunction(void (*IF)()) {
  RegisterHandler();
  InterruptFunction = IF;
  LeaveCriticalSection(&CriticalSection);
}


/// AddSignalHandler - Add a function to be called when a signal is delivered
/// to the process.  The handler can have a cookie passed to it to identify
/// what instance of the handler it is.
void llvm::sys::AddSignalHandler(void (*FnPtr)(void *), void *Cookie) {
  if (CallBacksToRun == 0)
    CallBacksToRun = new std::vector<std::pair<void(*)(void*), void*> >();
  CallBacksToRun->push_back(std::make_pair(FnPtr, Cookie));
  RegisterHandler();
  LeaveCriticalSection(&CriticalSection);
}

static void Cleanup() {
  if (CleanupExecuted)
    return;

  EnterCriticalSection(&CriticalSection);

  // Prevent other thread from registering new files and directories for
  // removal, should we be executing because of the console handler callback.
  CleanupExecuted = true;

  // FIXME: open files cannot be deleted.

  if (FilesToRemove != NULL)
    while (!FilesToRemove->empty()) {
      llvm::sys::fs::remove(FilesToRemove->back());
      FilesToRemove->pop_back();
    }

  if (CallBacksToRun)
    for (auto &I : *CallBacksToRun)
      I.first(I.second);

  LeaveCriticalSection(&CriticalSection);
}

void llvm::sys::RunInterruptHandlers() {
  // The interrupt handler may be called from an interrupt, but it may also be
  // called manually (such as the case of report_fatal_error with no registered
  // error handler). We must ensure that the critical section is properly
  // initialized.
  InitializeThreading();
  Cleanup();
}

static LONG WINAPI LLVMUnhandledExceptionFilter(LPEXCEPTION_POINTERS ep) {
  Cleanup();

  // Initialize the STACKFRAME structure.
  STACKFRAME64 StackFrame = {};

#if defined(_M_X64)
  StackFrame.AddrPC.Offset = ep->ContextRecord->Rip;
  StackFrame.AddrPC.Mode = AddrModeFlat;
  StackFrame.AddrStack.Offset = ep->ContextRecord->Rsp;
  StackFrame.AddrStack.Mode = AddrModeFlat;
  StackFrame.AddrFrame.Offset = ep->ContextRecord->Rbp;
  StackFrame.AddrFrame.Mode = AddrModeFlat;
#elif defined(_M_IX86)
  StackFrame.AddrPC.Offset = ep->ContextRecord->Eip;
  StackFrame.AddrPC.Mode = AddrModeFlat;
  StackFrame.AddrStack.Offset = ep->ContextRecord->Esp;
  StackFrame.AddrStack.Mode = AddrModeFlat;
  StackFrame.AddrFrame.Offset = ep->ContextRecord->Ebp;
  StackFrame.AddrFrame.Mode = AddrModeFlat;
#endif

  HANDLE hProcess = GetCurrentProcess();
  HANDLE hThread = GetCurrentThread();
  PrintStackTraceForThread(llvm::errs(), hProcess, hThread, StackFrame,
                           ep->ContextRecord);

  _exit(ep->ExceptionRecord->ExceptionCode);
}

static BOOL WINAPI LLVMConsoleCtrlHandler(DWORD dwCtrlType) {
  // We are running in our very own thread, courtesy of Windows.
  EnterCriticalSection(&CriticalSection);
  Cleanup();

  // If an interrupt function has been set, go and run one it; otherwise,
  // the process dies.
  void (*IF)() = InterruptFunction;
  InterruptFunction = 0;      // Don't run it on another CTRL-C.

  if (IF) {
    // Note: if the interrupt function throws an exception, there is nothing
    // to catch it in this thread so it will kill the process.
    IF();                     // Run it now.
    LeaveCriticalSection(&CriticalSection);
    return TRUE;              // Don't kill the process.
  }

  // Allow normal processing to take place; i.e., the process dies.
  LeaveCriticalSection(&CriticalSection);
  return FALSE;
}

#if __MINGW32__
 // We turned these warnings off for this file so that MinGW-g++ doesn't
 // complain about the ll format specifiers used.  Now we are turning the
 // warnings back on.  If MinGW starts to support diagnostic stacks, we can
 // replace this with a pop.
 #pragma GCC diagnostic warning "-Wformat"
 #pragma GCC diagnostic warning "-Wformat-extra-args"
#endif

#else

namespace llvm {
namespace sys {

bool RemoveFileOnSignal(StringRef Filename, std::string* ErrMsg) { return false; }
void DontRemoveFileOnSignal(StringRef Filename) { }
void PrintStackTraceOnErrorSignal(bool) { }
void PrintStackTrace(FILE *) { }
void SetInterruptFunction(void (*IF)()) { }
void AddSignalHandler(void (*FnPtr)(void *), void *Cookie) { }
void RunInterruptHandlers() { }

} }

#endif // HLSL Change - disable signal processing
