//===- lib/Support/ErrorHandling.cpp - Callbacks for errors ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines an API used to indicate fatal error conditions.  Non-fatal
// errors (most of them) should be handled through LLVMContext.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/ErrorHandling.h"
#include "llvm-c/Core.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Config/config.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Mutex.h"
#include "llvm/Support/MutexGuard.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/Threading.h"
#include "llvm/Support/WindowsError.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cstdlib>

#ifdef _WIN32
#include "windows.h"  // HLSL Change
#endif
#include "dxc/Support/exception.h"  // HLSL Change

#if defined(HAVE_UNISTD_H)
# include <unistd.h>
#endif
#if defined(_MSC_VER)
# include <io.h>
# include <fcntl.h>
#endif

using namespace llvm;

// HLSL Change - mark these as thread_local
thread_local static fatal_error_handler_t ErrorHandler = nullptr;
thread_local static void *ErrorHandlerUserData = nullptr;

// HLSL Change - no mutex needed, handlers are thread_local
//static ManagedStatic<sys::Mutex> ErrorHandlerMutex;

void llvm::install_fatal_error_handler(fatal_error_handler_t handler,
                                       void *user_data) {
  // llvm::MutexGuard Lock(*ErrorHandlerMutex); // HLSL Change - ErrorHandler and user data already thread-local
  assert(!ErrorHandler && "Error handler already registered!\n");
  ErrorHandler = handler;
  ErrorHandlerUserData = user_data;
}

void llvm::remove_fatal_error_handler() {
  // llvm::MutexGuard Lock(*ErrorHandlerMutex); // HLSL Change - ErrorHandler and user data already thread-local
  ErrorHandler = nullptr;
  ErrorHandlerUserData = nullptr;
}

void llvm::report_fatal_error(const char *Reason, bool GenCrashDiag) {
  report_fatal_error(Twine(Reason), GenCrashDiag);
}

void llvm::report_fatal_error(const std::string &Reason, bool GenCrashDiag) {
  report_fatal_error(Twine(Reason), GenCrashDiag);
}

void llvm::report_fatal_error(StringRef Reason, bool GenCrashDiag) {
  report_fatal_error(Twine(Reason), GenCrashDiag);
}

void llvm::report_fatal_error(const Twine &Reason, bool GenCrashDiag) {
  llvm::fatal_error_handler_t handler = nullptr;
  void* handlerData = nullptr;
  {
    // Only acquire the mutex while reading the handler, so as not to invoke a
    // user-supplied callback under a lock.
    // llvm::MutexGuard Lock(*ErrorHandlerMutex); // HLSL Change - ErrorHandler and user data already thread-local
    handler = ErrorHandler;
    handlerData = ErrorHandlerUserData;
  }

#ifndef LLVM_ON_WIN32 // HLSL Change - unwind if necessary, but don't terminate the process
  if (handler) {
    handler(handlerData, Reason.str(), GenCrashDiag);
  } else {
    // Blast the result out to stderr.  We don't try hard to make sure this
    // succeeds (e.g. handling EINTR) and we can't use errs() here because
    // raw ostreams can call report_fatal_error.
    SmallVector<char, 64> Buffer;
    raw_svector_ostream OS(Buffer);
    OS << "LLVM ERROR: " << Reason << "\n";
    StringRef MessageStr = OS.str();
    ssize_t written = ::write(2, MessageStr.data(), MessageStr.size());
    (void)written; // If something went wrong, we deliberately just give up.
  }

  // If we reached here, we are failing ungracefully. Run the interrupt handlers
  // to make sure any special cleanups get done, in particular that we remove
  // files registered with RemoveFileOnSignal.
  sys::RunInterruptHandlers();

  exit(1);
#else
  if (handler) {
    handler(handlerData, Reason.str(), GenCrashDiag);
  }

  throw hlsl::Exception(DXC_E_LLVM_FATAL_ERROR, std::string("LLVM ERROR: ") + Reason.str() + "\n");
#endif
}

void llvm::llvm_unreachable_internal(const char *msg, const char *file,
                                     unsigned line) {
  // This code intentionally doesn't call the ErrorHandler callback, because
  // llvm_unreachable is intended to be used to indicate "impossible"
  // situations, and not legitimate runtime errors.
  // HLSL Change - collect full message in string
  SmallVector<char, 64> Buffer;
  raw_svector_ostream OS(Buffer);
  if (msg)
    OS << msg << "\n";
  OS << "UNREACHABLE executed";
  if (file)
    OS << " at " << file << ":" << line;
  OS << "!\n";
#ifndef LLVM_ON_WIN32 // HLSL Change - unwind if necessary, but don't terminate the process
  dbgs() << OS.str();
  abort();
#else
  throw hlsl::Exception(DXC_E_LLVM_UNREACHABLE, OS.str());
#endif
}

void llvm::llvm_cast_assert_internal(const char *func) {
  throw hlsl::Exception(DXC_E_LLVM_CAST_ERROR, std::string(func) + "<X>() argument of incompatible type!\n");
}

static void bindingsErrorHandler(void *user_data, const std::string& reason,
                                 bool gen_crash_diag) {
  LLVMFatalErrorHandler handler =
      LLVM_EXTENSION reinterpret_cast<LLVMFatalErrorHandler>(user_data);
  handler(reason.c_str());
}

void LLVMInstallFatalErrorHandler(LLVMFatalErrorHandler Handler) {
  install_fatal_error_handler(bindingsErrorHandler,
                              LLVM_EXTENSION reinterpret_cast<void *>(Handler));
}

void LLVMResetFatalErrorHandler() {
  remove_fatal_error_handler();
}

#ifdef LLVM_ON_WIN32

#include <winerror.h>

// I'd rather not double the line count of the following.
#define MAP_ERR_TO_COND(x, y)                                                  \
  case x:                                                                      \
    return make_error_code(errc::y)

std::error_code llvm::mapWindowsError(unsigned EV) {
  switch (EV) {
    MAP_ERR_TO_COND(ERROR_ACCESS_DENIED, permission_denied);
    MAP_ERR_TO_COND(ERROR_ALREADY_EXISTS, file_exists);
    MAP_ERR_TO_COND(ERROR_BAD_UNIT, no_such_device);
    MAP_ERR_TO_COND(ERROR_BUFFER_OVERFLOW, filename_too_long);
    MAP_ERR_TO_COND(ERROR_BUSY, device_or_resource_busy);
    MAP_ERR_TO_COND(ERROR_BUSY_DRIVE, device_or_resource_busy);
    MAP_ERR_TO_COND(ERROR_CANNOT_MAKE, permission_denied);
    MAP_ERR_TO_COND(ERROR_CANTOPEN, io_error);
    MAP_ERR_TO_COND(ERROR_CANTREAD, io_error);
    MAP_ERR_TO_COND(ERROR_CANTWRITE, io_error);
    MAP_ERR_TO_COND(ERROR_CURRENT_DIRECTORY, permission_denied);
    MAP_ERR_TO_COND(ERROR_DEV_NOT_EXIST, no_such_device);
    MAP_ERR_TO_COND(ERROR_DEVICE_IN_USE, device_or_resource_busy);
    MAP_ERR_TO_COND(ERROR_DIR_NOT_EMPTY, directory_not_empty);
    MAP_ERR_TO_COND(ERROR_DIRECTORY, invalid_argument);
    MAP_ERR_TO_COND(ERROR_DISK_FULL, no_space_on_device);
    MAP_ERR_TO_COND(ERROR_FILE_EXISTS, file_exists);
    MAP_ERR_TO_COND(ERROR_FILE_NOT_FOUND, no_such_file_or_directory);
    MAP_ERR_TO_COND(ERROR_HANDLE_DISK_FULL, no_space_on_device);
    MAP_ERR_TO_COND(ERROR_INVALID_ACCESS, permission_denied);
    MAP_ERR_TO_COND(ERROR_INVALID_DRIVE, no_such_device);
    MAP_ERR_TO_COND(ERROR_INVALID_FUNCTION, function_not_supported);
    MAP_ERR_TO_COND(ERROR_INVALID_HANDLE, invalid_argument);
    MAP_ERR_TO_COND(ERROR_INVALID_NAME, invalid_argument);
    MAP_ERR_TO_COND(ERROR_LOCK_VIOLATION, no_lock_available);
    MAP_ERR_TO_COND(ERROR_LOCKED, no_lock_available);
    MAP_ERR_TO_COND(ERROR_NEGATIVE_SEEK, invalid_argument);
    MAP_ERR_TO_COND(ERROR_NOACCESS, permission_denied);
    MAP_ERR_TO_COND(ERROR_NOT_ENOUGH_MEMORY, not_enough_memory);
    MAP_ERR_TO_COND(ERROR_NOT_READY, resource_unavailable_try_again);
    MAP_ERR_TO_COND(ERROR_OPEN_FAILED, io_error);
    MAP_ERR_TO_COND(ERROR_OPEN_FILES, device_or_resource_busy);
    MAP_ERR_TO_COND(ERROR_OUTOFMEMORY, not_enough_memory);
    MAP_ERR_TO_COND(ERROR_PATH_NOT_FOUND, no_such_file_or_directory);
    MAP_ERR_TO_COND(ERROR_BAD_NETPATH, no_such_file_or_directory);
    MAP_ERR_TO_COND(ERROR_READ_FAULT, io_error);
    MAP_ERR_TO_COND(ERROR_RETRY, resource_unavailable_try_again);
    MAP_ERR_TO_COND(ERROR_SEEK, io_error);
    MAP_ERR_TO_COND(ERROR_SHARING_VIOLATION, permission_denied);
    MAP_ERR_TO_COND(ERROR_TOO_MANY_OPEN_FILES, too_many_files_open);
    MAP_ERR_TO_COND(ERROR_WRITE_FAULT, io_error);
    MAP_ERR_TO_COND(ERROR_WRITE_PROTECT, permission_denied);
    MAP_ERR_TO_COND(WSAEACCES, permission_denied);
    MAP_ERR_TO_COND(WSAEBADF, bad_file_descriptor);
    MAP_ERR_TO_COND(WSAEFAULT, bad_address);
    MAP_ERR_TO_COND(WSAEINTR, interrupted);
    MAP_ERR_TO_COND(WSAEINVAL, invalid_argument);
    MAP_ERR_TO_COND(WSAEMFILE, too_many_files_open);
    MAP_ERR_TO_COND(WSAENAMETOOLONG, filename_too_long);
  default:
    return std::error_code(EV, std::system_category());
  }
}

#endif
