//===--- CrashRecoveryContext.cpp - Crash Recovery ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/CrashRecoveryContext.h"
#include "llvm/Config/config.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Mutex.h"
#include "llvm/Support/ThreadLocal.h"
#include <setjmp.h>
using namespace llvm;

namespace {

struct CrashRecoveryContextImpl;

static ManagedStatic<
    sys::ThreadLocal<const CrashRecoveryContextImpl> > CurrentContext;

struct CrashRecoveryContextImpl {
  CrashRecoveryContext *CRC;
  std::string Backtrace;
  ::jmp_buf JumpBuffer;
  volatile unsigned Failed : 1;
  unsigned SwitchedThread : 1;

public:
  CrashRecoveryContextImpl(CrashRecoveryContext *CRC) : CRC(CRC),
                                                        Failed(false),
                                                        SwitchedThread(false) {
    CurrentContext->set(this);
  }
  ~CrashRecoveryContextImpl() {
    if (!SwitchedThread)
      CurrentContext->erase();
  }

  /// \brief Called when the separate crash-recovery thread was finished, to
  /// indicate that we don't need to clear the thread-local CurrentContext.
  void setSwitchedThread() { SwitchedThread = true; }

  void HandleCrash() {
    // Eliminate the current context entry, to avoid re-entering in case the
    // cleanup code crashes.
    CurrentContext->erase();

    assert(!Failed && "Crash recovery context already failed!");
    Failed = true;

    // FIXME: Stash the backtrace.

    // Jump back to the RunSafely we were called under.
    longjmp(JumpBuffer, 1);
  }
};

}

static ManagedStatic<sys::Mutex> gCrashRecoveryContextMutex;
static bool gCrashRecoveryEnabled = false;

static ManagedStatic<sys::ThreadLocal<const CrashRecoveryContextCleanup> >
       tlIsRecoveringFromCrash;

CrashRecoveryContextCleanup::~CrashRecoveryContextCleanup() {}

CrashRecoveryContext::~CrashRecoveryContext() {
  // Reclaim registered resources.
  CrashRecoveryContextCleanup *i = head;
  tlIsRecoveringFromCrash->set(head);
  while (i) {
    CrashRecoveryContextCleanup *tmp = i;
    i = tmp->next;
    tmp->cleanupFired = true;
    tmp->recoverResources();
    delete tmp;
  }
  tlIsRecoveringFromCrash->erase();
  
  CrashRecoveryContextImpl *CRCI = (CrashRecoveryContextImpl *) Impl;
  delete CRCI;
}

bool CrashRecoveryContext::isRecoveringFromCrash() {
  return tlIsRecoveringFromCrash->get() != nullptr;
}

CrashRecoveryContext *CrashRecoveryContext::GetCurrent() {
  if (!gCrashRecoveryEnabled)
    return nullptr;

  const CrashRecoveryContextImpl *CRCI = CurrentContext->get();
  if (!CRCI)
    return nullptr;

  return CRCI->CRC;
}

void CrashRecoveryContext::registerCleanup(CrashRecoveryContextCleanup *cleanup)
{
  if (!cleanup)
    return;
  if (head)
    head->prev = cleanup;
  cleanup->next = head;
  head = cleanup;
}

void
CrashRecoveryContext::unregisterCleanup(CrashRecoveryContextCleanup *cleanup) {
  if (!cleanup)
    return;
  if (cleanup == head) {
    head = cleanup->next;
    if (head)
      head->prev = nullptr;
  }
  else {
    cleanup->prev->next = cleanup->next;
    if (cleanup->next)
      cleanup->next->prev = cleanup->prev;
  }
  delete cleanup;
}

#ifdef LLVM_ON_WIN32

#include "Windows/WindowsSupport.h"

// On Windows, we can make use of vectored exception handling to
// catch most crashing situations.  Note that this does mean
// we will be alerted of exceptions *before* structured exception
// handling has the opportunity to catch it.  But that isn't likely
// to cause problems because nowhere in the project is SEH being
// used.
//
// Vectored exception handling is built on top of SEH, and so it
// works on a per-thread basis.
//
// The vectored exception handler functionality was added in Windows
// XP, so if support for older versions of Windows is required,
// it will have to be added.
//
// If we want to support as far back as Win2k, we could use the
// SetUnhandledExceptionFilter API, but there's a risk of that
// being entirely overwritten (it's not a chain).

static LONG CALLBACK ExceptionHandler(PEXCEPTION_POINTERS ExceptionInfo)
{
  // Lookup the current thread local recovery object.
  const CrashRecoveryContextImpl *CRCI = CurrentContext->get();

  if (!CRCI) {
    // Something has gone horribly wrong, so let's just tell everyone
    // to keep searching
    CrashRecoveryContext::Disable();
    return EXCEPTION_CONTINUE_SEARCH;
  }

  // TODO: We can capture the stack backtrace here and store it on the
  // implementation if we so choose.

  // Handle the crash
  const_cast<CrashRecoveryContextImpl*>(CRCI)->HandleCrash();

  // Note that we don't actually get here because HandleCrash calls
  // longjmp, which means the HandleCrash function never returns.
  llvm_unreachable("Handled the crash, should have longjmp'ed out of here");
}

// Because the Enable and Disable calls are static, it means that
// there may not actually be an Impl available, or even a current
// CrashRecoveryContext at all.  So we make use of a thread-local
// exception table.  The handles contained in here will either be
// non-NULL, valid VEH handles, or NULL.
static sys::ThreadLocal<const void> sCurrentExceptionHandle;

void CrashRecoveryContext::Enable() {
  sys::ScopedLock L(*gCrashRecoveryContextMutex);

  if (gCrashRecoveryEnabled)
    return;

  gCrashRecoveryEnabled = true;

  // We can set up vectored exception handling now.  We will install our
  // handler as the front of the list, though there's no assurances that
  // it will remain at the front (another call could install itself before
  // our handler).  This 1) isn't likely, and 2) shouldn't cause problems.
  PVOID handle = ::AddVectoredExceptionHandler(1, ExceptionHandler);
  sCurrentExceptionHandle.set(handle);
}

void CrashRecoveryContext::Disable() {
  sys::ScopedLock L(*gCrashRecoveryContextMutex);

  if (!gCrashRecoveryEnabled)
    return;

  gCrashRecoveryEnabled = false;

  PVOID currentHandle = const_cast<PVOID>(sCurrentExceptionHandle.get());
  if (currentHandle) {
    // Now we can remove the vectored exception handler from the chain
    ::RemoveVectoredExceptionHandler(currentHandle);

    // Reset the handle in our thread-local set.
    sCurrentExceptionHandle.set(NULL);
  }
}

#else

// Generic POSIX implementation.
//
// This implementation relies on synchronous signals being delivered to the
// current thread. We use a thread local object to keep track of the active
// crash recovery context, and install signal handlers to invoke HandleCrash on
// the active object.
//
// This implementation does not to attempt to chain signal handlers in any
// reliable fashion -- if we get a signal outside of a crash recovery context we
// simply disable crash recovery and raise the signal again.

#include <signal.h>

static const int Signals[] =
    { SIGABRT, SIGBUS, SIGFPE, SIGILL, SIGSEGV, SIGTRAP };
static const unsigned NumSignals = sizeof(Signals) / sizeof(Signals[0]);
static struct sigaction PrevActions[NumSignals];

static void CrashRecoverySignalHandler(int Signal) {
  // Lookup the current thread local recovery object.
  const CrashRecoveryContextImpl *CRCI = CurrentContext->get();

  if (!CRCI) {
    // We didn't find a crash recovery context -- this means either we got a
    // signal on a thread we didn't expect it on, the application got a signal
    // outside of a crash recovery context, or something else went horribly
    // wrong.
    //
    // Disable crash recovery and raise the signal again. The assumption here is
    // that the enclosing application will terminate soon, and we won't want to
    // attempt crash recovery again.
    //
    // This call of Disable isn't thread safe, but it doesn't actually matter.
    CrashRecoveryContext::Disable();
    raise(Signal);

    // The signal will be thrown once the signal mask is restored.
    return;
  }

  // Unblock the signal we received.
  sigset_t SigMask;
  sigemptyset(&SigMask);
  sigaddset(&SigMask, Signal);
  sigprocmask(SIG_UNBLOCK, &SigMask, nullptr);

  if (CRCI)
    const_cast<CrashRecoveryContextImpl*>(CRCI)->HandleCrash();
}

void CrashRecoveryContext::Enable() {
  sys::ScopedLock L(*gCrashRecoveryContextMutex);

  if (gCrashRecoveryEnabled)
    return;

  gCrashRecoveryEnabled = true;

  // Setup the signal handler.
  struct sigaction Handler;
  Handler.sa_handler = CrashRecoverySignalHandler;
  Handler.sa_flags = 0;
  sigemptyset(&Handler.sa_mask);

  for (unsigned i = 0; i != NumSignals; ++i) {
    sigaction(Signals[i], &Handler, &PrevActions[i]);
  }
}

void CrashRecoveryContext::Disable() {
  sys::ScopedLock L(*gCrashRecoveryContextMutex);

  if (!gCrashRecoveryEnabled)
    return;

  gCrashRecoveryEnabled = false;

  // Restore the previous signal handlers.
  for (unsigned i = 0; i != NumSignals; ++i)
    sigaction(Signals[i], &PrevActions[i], nullptr);
}

#endif

bool CrashRecoveryContext::RunSafely(function_ref<void()> Fn) {
  // If crash recovery is disabled, do nothing.
  if (gCrashRecoveryEnabled) {
    assert(!Impl && "Crash recovery context already initialized!");
    CrashRecoveryContextImpl *CRCI = new CrashRecoveryContextImpl(this);
    Impl = CRCI;

    if (setjmp(CRCI->JumpBuffer) != 0) {
      return false;
    }
  }

  Fn();
  return true;
}

void CrashRecoveryContext::HandleCrash() {
  CrashRecoveryContextImpl *CRCI = (CrashRecoveryContextImpl *) Impl;
  assert(CRCI && "Crash recovery context never initialized!");
  CRCI->HandleCrash();
}

const std::string &CrashRecoveryContext::getBacktrace() const {
  CrashRecoveryContextImpl *CRC = (CrashRecoveryContextImpl *) Impl;
  assert(CRC && "Crash recovery context never initialized!");
  assert(CRC->Failed && "No crash was detected!");
  return CRC->Backtrace;
}

// FIXME: Portability.
static void setThreadBackgroundPriority() {
#ifdef __APPLE__
  setpriority(PRIO_DARWIN_THREAD, 0, PRIO_DARWIN_BG);
#endif
}

static bool hasThreadBackgroundPriority() {
#ifdef __APPLE__
  return getpriority(PRIO_DARWIN_THREAD, 0) == 1;
#else
  return false;
#endif
}

namespace {
struct RunSafelyOnThreadInfo {
  function_ref<void()> Fn;
  CrashRecoveryContext *CRC;
  bool UseBackgroundPriority;
  bool Result;
};
}

static void RunSafelyOnThread_Dispatch(void *UserData) {
  RunSafelyOnThreadInfo *Info =
    reinterpret_cast<RunSafelyOnThreadInfo*>(UserData);

  if (Info->UseBackgroundPriority)
    setThreadBackgroundPriority();

  Info->Result = Info->CRC->RunSafely(Info->Fn);
}
bool CrashRecoveryContext::RunSafelyOnThread(function_ref<void()> Fn,
                                             unsigned RequestedStackSize) {
  bool UseBackgroundPriority = hasThreadBackgroundPriority();
  RunSafelyOnThreadInfo Info = { Fn, this, UseBackgroundPriority, false };
  llvm_execute_on_thread(RunSafelyOnThread_Dispatch, &Info, RequestedStackSize);
  if (CrashRecoveryContextImpl *CRC = (CrashRecoveryContextImpl *)Impl)
    CRC->setSwitchedThread();
  return Info.Result;
}
