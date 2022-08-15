//===- Signals.cpp - Generic Unix Signals Implementation -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines some helpful functions for dealing with the possibility of
// Unix signals occurring while your program is running.
//
//===----------------------------------------------------------------------===//

#include "Unix.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Mutex.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/UniqueLock.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <string>
#include <vector>
#if HAVE_EXECINFO_H
# include <execinfo.h>         // For backtrace().
#endif
#if HAVE_SIGNAL_H
#include <signal.h>
#endif
#if HAVE_SYS_STAT_H
#include <sys/stat.h>
#endif
#if HAVE_CXXABI_H
#include <cxxabi.h>
#endif
#if HAVE_DLFCN_H
#include <dlfcn.h>
#endif
#if HAVE_MACH_MACH_H
#include <mach/mach.h>
#endif
#if HAVE_LINK_H
#include <link.h>
#endif

using namespace llvm;

static RETSIGTYPE SignalHandler(int Sig);  // defined below.

static ManagedStatic<SmartMutex<true> > SignalsMutex;

/// InterruptFunction - The function to call if ctrl-c is pressed.
static void (*InterruptFunction)() = nullptr;

static ManagedStatic<std::vector<std::string>> FilesToRemove;
static ManagedStatic<std::vector<std::pair<void (*)(void *), void *>>>
    CallBacksToRun;

// IntSigs - Signals that represent requested termination. There's no bug
// or failure, or if there is, it's not our direct responsibility. For whatever
// reason, our continued execution is no longer desirable.
static const int IntSigs[] = {
  SIGHUP, SIGINT, SIGPIPE, SIGTERM, SIGUSR1, SIGUSR2
};

// KillSigs - Signals that represent that we have a bug, and our prompt
// termination has been ordered.
static const int KillSigs[] = {
  SIGILL, SIGTRAP, SIGABRT, SIGFPE, SIGBUS, SIGSEGV, SIGQUIT
#ifdef SIGSYS
  , SIGSYS
#endif
#ifdef SIGXCPU
  , SIGXCPU
#endif
#ifdef SIGXFSZ
  , SIGXFSZ
#endif
#ifdef SIGEMT
  , SIGEMT
#endif
};

static unsigned NumRegisteredSignals = 0;
static struct {
  struct sigaction SA;
  int SigNo;
} RegisteredSignalInfo[(sizeof(IntSigs)+sizeof(KillSigs))/sizeof(KillSigs[0])];


static void RegisterHandler(int Signal) {
  assert(NumRegisteredSignals <
         sizeof(RegisteredSignalInfo)/sizeof(RegisteredSignalInfo[0]) &&
         "Out of space for signal handlers!");

  struct sigaction NewHandler;

  NewHandler.sa_handler = SignalHandler;
  NewHandler.sa_flags = SA_NODEFER|SA_RESETHAND;
  sigemptyset(&NewHandler.sa_mask);

  // Install the new handler, save the old one in RegisteredSignalInfo.
  sigaction(Signal, &NewHandler,
            &RegisteredSignalInfo[NumRegisteredSignals].SA);
  RegisteredSignalInfo[NumRegisteredSignals].SigNo = Signal;
  ++NumRegisteredSignals;
}

static void RegisterHandlers() {
  // We need to dereference the signals mutex during handler registration so
  // that we force its construction. This is to prevent the first use being
  // during handling an actual signal because you can't safely call new in a
  // signal handler.
  *SignalsMutex;
  
  // If the handlers are already registered, we're done.
  if (NumRegisteredSignals != 0) return;

  for (auto S : IntSigs) RegisterHandler(S);
  for (auto S : KillSigs) RegisterHandler(S);
}

static void UnregisterHandlers() {
  // Restore all of the signal handlers to how they were before we showed up.
  for (unsigned i = 0, e = NumRegisteredSignals; i != e; ++i)
    sigaction(RegisteredSignalInfo[i].SigNo,
              &RegisteredSignalInfo[i].SA, nullptr);
  NumRegisteredSignals = 0;
}


/// RemoveFilesToRemove - Process the FilesToRemove list. This function
/// should be called with the SignalsMutex lock held.
/// NB: This must be an async signal safe function. It cannot allocate or free
/// memory, even in debug builds.
static void RemoveFilesToRemove() {
  // Avoid constructing ManagedStatic in the signal handler.
  // If FilesToRemove is not constructed, there are no files to remove.
  if (!FilesToRemove.isConstructed())
    return;

  // We avoid iterators in case of debug iterators that allocate or release
  // memory.
  std::vector<std::string>& FilesToRemoveRef = *FilesToRemove;
  for (unsigned i = 0, e = FilesToRemoveRef.size(); i != e; ++i) {
    // We rely on a std::string implementation for which repeated calls to
    // 'c_str()' don't allocate memory. We pre-call 'c_str()' on all of these
    // strings to try to ensure this is safe.
    const char *path = FilesToRemoveRef[i].c_str();

    // Get the status so we can determine if it's a file or directory. If we
    // can't stat the file, ignore it.
    struct stat buf;
    if (stat(path, &buf) != 0)
      continue;

    // If this is not a regular file, ignore it. We want to prevent removal of
    // special files like /dev/null, even if the compiler is being run with the
    // super-user permissions.
    if (!S_ISREG(buf.st_mode))
      continue;
  
    // Otherwise, remove the file. We ignore any errors here as there is nothing
    // else we can do.
    unlink(path);
  }
}

// SignalHandler - The signal handler that runs.
static RETSIGTYPE SignalHandler(int Sig) {
  // Restore the signal behavior to default, so that the program actually
  // crashes when we return and the signal reissues.  This also ensures that if
  // we crash in our signal handler that the program will terminate immediately
  // instead of recursing in the signal handler.
  UnregisterHandlers();

  // Unmask all potentially blocked kill signals.
  sigset_t SigMask;
  sigfillset(&SigMask);
  sigprocmask(SIG_UNBLOCK, &SigMask, nullptr);

  {
    unique_lock<SmartMutex<true>> Guard(*SignalsMutex);
    RemoveFilesToRemove();

    if (std::find(std::begin(IntSigs), std::end(IntSigs), Sig)
        != std::end(IntSigs)) {
      if (InterruptFunction) {
        void (*IF)() = InterruptFunction;
        Guard.unlock();
        InterruptFunction = nullptr;
        IF();        // run the interrupt function.
        return;
      }

      Guard.unlock();
      raise(Sig);   // Execute the default handler.
      return;
   }
  }

  // Otherwise if it is a fault (like SEGV) run any handler.
  if (CallBacksToRun.isConstructed()) {
    auto &CallBacksToRunRef = *CallBacksToRun;
    for (unsigned i = 0, e = CallBacksToRun->size(); i != e; ++i)
      CallBacksToRunRef[i].first(CallBacksToRunRef[i].second);
  }

#ifdef __s390__
  // On S/390, certain signals are delivered with PSW Address pointing to
  // *after* the faulting instruction.  Simply returning from the signal
  // handler would continue execution after that point, instead of
  // re-raising the signal.  Raise the signal manually in those cases.
  if (Sig == SIGILL || Sig == SIGFPE || Sig == SIGTRAP)
    raise(Sig);
#endif
}

void llvm::sys::RunInterruptHandlers() {
  sys::SmartScopedLock<true> Guard(*SignalsMutex);
  RemoveFilesToRemove();
}

void llvm::sys::SetInterruptFunction(void (*IF)()) {
  {
    sys::SmartScopedLock<true> Guard(*SignalsMutex);
    InterruptFunction = IF;
  }
  RegisterHandlers();
}

// RemoveFileOnSignal - The public API
bool llvm::sys::RemoveFileOnSignal(StringRef Filename,
                                   std::string* ErrMsg) {
  {
    sys::SmartScopedLock<true> Guard(*SignalsMutex);
    std::vector<std::string>& FilesToRemoveRef = *FilesToRemove;
    std::string *OldPtr =
        FilesToRemoveRef.empty() ? nullptr : &FilesToRemoveRef[0];
    FilesToRemoveRef.push_back(Filename);

    // We want to call 'c_str()' on every std::string in this vector so that if
    // the underlying implementation requires a re-allocation, it happens here
    // rather than inside of the signal handler. If we see the vector grow, we
    // have to call it on every entry. If it remains in place, we only need to
    // call it on the latest one.
    if (OldPtr == &FilesToRemoveRef[0])
      FilesToRemoveRef.back().c_str();
    else
      for (unsigned i = 0, e = FilesToRemoveRef.size(); i != e; ++i)
        FilesToRemoveRef[i].c_str();
  }

  RegisterHandlers();
  return false;
}

// DontRemoveFileOnSignal - The public API
void llvm::sys::DontRemoveFileOnSignal(StringRef Filename) {
  sys::SmartScopedLock<true> Guard(*SignalsMutex);
  std::vector<std::string>::reverse_iterator RI =
    std::find(FilesToRemove->rbegin(), FilesToRemove->rend(), Filename);
  std::vector<std::string>::iterator I = FilesToRemove->end();
  if (RI != FilesToRemove->rend())
    I = FilesToRemove->erase(RI.base()-1);

  // We need to call c_str() on every element which would have been moved by
  // the erase. These elements, in a C++98 implementation where c_str()
  // requires a reallocation on the first call may have had the call to c_str()
  // made on insertion become invalid by being copied down an element.
  for (std::vector<std::string>::iterator E = FilesToRemove->end(); I != E; ++I)
    I->c_str();
}

/// AddSignalHandler - Add a function to be called when a signal is delivered
/// to the process.  The handler can have a cookie passed to it to identify
/// what instance of the handler it is.
void llvm::sys::AddSignalHandler(void (*FnPtr)(void *), void *Cookie) {
  CallBacksToRun->push_back(std::make_pair(FnPtr, Cookie));
  RegisterHandlers();
}

#if defined(HAVE_BACKTRACE) && defined(ENABLE_BACKTRACES)

#if HAVE_LINK_H && (defined(__linux__) || defined(__FreeBSD__) ||              \
                    defined(__FreeBSD_kernel__) || defined(__NetBSD__))
struct DlIteratePhdrData {
  void **StackTrace;
  int depth;
  bool first;
  const char **modules;
  intptr_t *offsets;
  const char *main_exec_name;
};

static int dl_iterate_phdr_cb(dl_phdr_info *info, size_t size, void *arg) {
  DlIteratePhdrData *data = (DlIteratePhdrData*)arg;
  const char *name = data->first ? data->main_exec_name : info->dlpi_name;
  data->first = false;
  for (int i = 0; i < info->dlpi_phnum; i++) {
    const auto *phdr = &info->dlpi_phdr[i];
    if (phdr->p_type != PT_LOAD)
      continue;
    intptr_t beg = info->dlpi_addr + phdr->p_vaddr;
    intptr_t end = beg + phdr->p_memsz;
    for (int j = 0; j < data->depth; j++) {
      if (data->modules[j])
        continue;
      intptr_t addr = (intptr_t)data->StackTrace[j];
      if (beg <= addr && addr < end) {
        data->modules[j] = name;
        data->offsets[j] = addr - info->dlpi_addr;
      }
    }
  }
  return 0;
}

static bool findModulesAndOffsets(void **StackTrace, int Depth,
                                  const char **Modules, intptr_t *Offsets,
                                  const char *MainExecutableName) {
  DlIteratePhdrData data = {StackTrace, Depth,   true,
                            Modules,    Offsets, MainExecutableName};
  dl_iterate_phdr(dl_iterate_phdr_cb, &data);
  return true;
}
#else
static bool findModulesAndOffsets(void **StackTrace, int Depth,
                                  const char **Modules, intptr_t *Offsets,
                                  const char *MainExecutableName) {
  return false;
}
#endif

static bool printSymbolizedStackTrace(void **StackTrace, int Depth,
                                      llvm::raw_ostream &OS) {
  // FIXME: Subtract necessary number from StackTrace entries to turn return addresses
  // into actual instruction addresses.
  // Use llvm-symbolizer tool to symbolize the stack traces.
  ErrorOr<std::string> LLVMSymbolizerPathOrErr =
      sys::findProgramByName("llvm-symbolizer");
  if (!LLVMSymbolizerPathOrErr)
    return false;
  const std::string &LLVMSymbolizerPath = *LLVMSymbolizerPathOrErr;
  // We don't know argv0 or the address of main() at this point, but try
  // to guess it anyway (it's possible on some platforms).
  std::string MainExecutableName = sys::fs::getMainExecutable(nullptr, nullptr);
  if (MainExecutableName.empty() ||
      MainExecutableName.find("llvm-symbolizer") != std::string::npos)
    return false;

  std::vector<const char *> Modules(Depth, nullptr);
  std::vector<intptr_t> Offsets(Depth, 0);
  if (!findModulesAndOffsets(StackTrace, Depth, Modules.data(), Offsets.data(),
                             MainExecutableName.c_str()))
    return false;
  int InputFD;
  SmallString<32> InputFile, OutputFile;
  sys::fs::createTemporaryFile("symbolizer-input", "", InputFD, InputFile);
  sys::fs::createTemporaryFile("symbolizer-output", "", OutputFile);
  FileRemover InputRemover(InputFile.c_str());
  FileRemover OutputRemover(OutputFile.c_str());

  {
    raw_fd_ostream Input(InputFD, true);
    for (int i = 0; i < Depth; i++) {
      if (Modules[i])
        Input << Modules[i] << " " << (void*)Offsets[i] << "\n";
    }
  }

  StringRef InputFileStr(InputFile);
  StringRef OutputFileStr(OutputFile);
  StringRef StderrFileStr;
  const StringRef *Redirects[] = {&InputFileStr, &OutputFileStr,
                                  &StderrFileStr};
  const char *Args[] = {"llvm-symbolizer", "--functions=linkage", "--inlining",
                        "--demangle", nullptr};
  int RunResult =
      sys::ExecuteAndWait(LLVMSymbolizerPath, Args, nullptr, Redirects);
  if (RunResult != 0)
    return false;

  auto OutputBuf = MemoryBuffer::getFile(OutputFile.c_str());
  if (!OutputBuf)
    return false;
  StringRef Output = OutputBuf.get()->getBuffer();
  SmallVector<StringRef, 32> Lines;
  Output.split(Lines, "\n");
  auto CurLine = Lines.begin();
  int frame_no = 0;
  for (int i = 0; i < Depth; i++) {
    if (!Modules[i]) {
      OS << format("#%d %p\n", frame_no++, StackTrace[i]);
      continue;
    }
    // Read pairs of lines (function name and file/line info) until we
    // encounter empty line.
    for (;;) {
      if (CurLine == Lines.end())
        return false;
      StringRef FunctionName = *CurLine++;
      if (FunctionName.empty())
        break;
      OS << format("#%d %p ", frame_no++, StackTrace[i]);
      if (!FunctionName.startswith("??"))
        OS << format("%s ", FunctionName.str().c_str());
      if (CurLine == Lines.end())
        return false;
      StringRef FileLineInfo = *CurLine++;
      if (!FileLineInfo.startswith("??"))
        OS << format("%s", FileLineInfo.str().c_str());
      else
        OS << format("(%s+%p)", Modules[i], (void *)Offsets[i]);
      OS << "\n";
    }
  }
  return true;
}
#endif // defined(HAVE_BACKTRACE) && defined(ENABLE_BACKTRACES)

// PrintStackTrace - In the case of a program crash or fault, print out a stack
// trace so that the user has an indication of why and where we died.
//
// On glibc systems we have the 'backtrace' function, which works nicely, but
// doesn't demangle symbols.
void llvm::sys::PrintStackTrace(raw_ostream &OS) {
#if defined(HAVE_BACKTRACE) && defined(ENABLE_BACKTRACES)
  static void* StackTrace[256];
  // Use backtrace() to output a backtrace on Linux systems with glibc.
  int depth = backtrace(StackTrace,
                        static_cast<int>(array_lengthof(StackTrace)));
  if (printSymbolizedStackTrace(StackTrace, depth, OS))
    return;
#if HAVE_DLFCN_H && __GNUG__
  int width = 0;
  for (int i = 0; i < depth; ++i) {
    Dl_info dlinfo;
    dladdr(StackTrace[i], &dlinfo);
    const char* name = strrchr(dlinfo.dli_fname, '/');

    int nwidth;
    if (!name) nwidth = strlen(dlinfo.dli_fname);
    else       nwidth = strlen(name) - 1;

    if (nwidth > width) width = nwidth;
  }

  for (int i = 0; i < depth; ++i) {
    Dl_info dlinfo;
    dladdr(StackTrace[i], &dlinfo);

    OS << format("%-2d", i);

    const char* name = strrchr(dlinfo.dli_fname, '/');
    if (!name) OS << format(" %-*s", width, dlinfo.dli_fname);
    else       OS << format(" %-*s", width, name+1);

    OS << format(" %#0*lx", (int)(sizeof(void*) * 2) + 2,
                 (unsigned long)StackTrace[i]);

    if (dlinfo.dli_sname != nullptr) {
      OS << ' ';
#  if HAVE_CXXABI_H
      int res;
      char* d = abi::__cxa_demangle(dlinfo.dli_sname, nullptr, nullptr, &res);
#  else
      char* d = NULL;
#  endif
      if (!d) OS << dlinfo.dli_sname;
      else    OS << d;
      free(d);

      // FIXME: When we move to C++11, use %t length modifier. It's not in
      // C++03 and causes gcc to issue warnings. Losing the upper 32 bits of
      // the stack offset for a stack dump isn't likely to cause any problems.
      OS << format(" + %u",(unsigned)((char*)StackTrace[i]-
                                      (char*)dlinfo.dli_saddr));
    }
    OS << '\n';
  }
#else
  backtrace_symbols_fd(StackTrace, depth, STDERR_FILENO);
#endif
#endif
}

static void PrintStackTraceSignalHandler(void *) {
  PrintStackTrace(llvm::errs());
}

void llvm::sys::DisableSystemDialogsOnCrash() {}

/// PrintStackTraceOnErrorSignal - When an error signal (such as SIGABRT or
/// SIGSEGV) is delivered to the process, print a stack trace and then exit.
void llvm::sys::PrintStackTraceOnErrorSignal(bool DisableCrashReporting) {
  AddSignalHandler(PrintStackTraceSignalHandler, nullptr);

#if defined(__APPLE__) && defined(ENABLE_CRASH_OVERRIDES)
  // Environment variable to disable any kind of crash dialog.
  if (DisableCrashReporting || getenv("LLVM_DISABLE_CRASH_REPORT")) {
    mach_port_t self = mach_task_self();

    exception_mask_t mask = EXC_MASK_CRASH;

    kern_return_t ret = task_set_exception_ports(self,
                             mask,
                             MACH_PORT_NULL,
                             EXCEPTION_STATE_IDENTITY | MACH_EXCEPTION_CODES,
                             THREAD_STATE_NONE);
    (void)ret;
  }
#endif
}


/***/

// On Darwin, raise sends a signal to the main thread instead of the current
// thread. This has the unfortunate effect that assert() and abort() will end up
// bypassing our crash recovery attempts. We work around this for anything in
// the same linkage unit by just defining our own versions of the assert handler
// and abort.

#if defined(__APPLE__) && defined(ENABLE_CRASH_OVERRIDES)

#include <signal.h>
#include <pthread.h>

int raise(int sig) {
  return pthread_kill(pthread_self(), sig);
}

void __assert_rtn(const char *func,
                  const char *file,
                  int line,
                  const char *expr) {
  if (func)
    fprintf(stderr, "Assertion failed: (%s), function %s, file %s, line %d.\n",
            expr, func, file, line);
  else
    fprintf(stderr, "Assertion failed: (%s), file %s, line %d.\n",
            expr, file, line);
  abort();
}

void abort() {
  raise(SIGABRT);
  usleep(1000);
  __builtin_trap();
}

#endif
