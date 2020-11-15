// Copyright 2015 The Crashpad Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "client/crashpad_client.h"

#include <windows.h>
#include <signal.h>
#include <stdint.h>
#include <string.h>

#include <memory>

#include "base/atomicops.h"
#include "base/logging.h"
#include "base/macros.h"
#include "base/scoped_generic.h"
#include "base/strings/string16.h"
#include "base/strings/stringprintf.h"
#include "base/strings/utf_string_conversions.h"
#include "base/synchronization/lock.h"
#include "util/file/file_io.h"
#include "util/misc/capture_context.h"
#include "util/misc/from_pointer_cast.h"
#include "util/misc/random_string.h"
#include "util/win/address_types.h"
#include "util/win/command_line.h"
#include "util/win/critical_section_with_debug_info.h"
#include "util/win/get_function.h"
#include "util/win/handle.h"
#include "util/win/initial_client_data.h"
#include "util/win/nt_internals.h"
#include "util/win/ntstatus_logging.h"
#include "util/win/process_info.h"
#include "util/win/registration_protocol_win.h"
#include "util/win/safe_terminate_process.h"
#include "util/win/scoped_process_suspend.h"
#include "util/win/termination_codes.h"
#include "util/win/xp_compat.h"

namespace crashpad {

namespace {

// This handle is never closed. This is used to signal to the server that a dump
// should be taken in the event of a crash.
HANDLE g_signal_exception = INVALID_HANDLE_VALUE;

// Where we store the exception information that the crash handler reads.
ExceptionInformation g_crash_exception_information;

// These handles are never closed. g_signal_non_crash_dump is used to signal to
// the server to take a dump (not due to an exception), and the server will
// signal g_non_crash_dump_done when the dump is completed.
HANDLE g_signal_non_crash_dump = INVALID_HANDLE_VALUE;
HANDLE g_non_crash_dump_done = INVALID_HANDLE_VALUE;

// Guards multiple simultaneous calls to DumpWithoutCrash(). This is leaked.
base::Lock* g_non_crash_dump_lock;

// Where we store a pointer to the context information when taking a non-crash
// dump.
ExceptionInformation g_non_crash_exception_information;

enum class StartupState : int {
  kNotReady = 0,   // This must be value 0 because it is the initial value of a
                   // global AtomicWord.
  kSucceeded = 1,  // The CreateProcess() for the handler succeeded.
  kFailed = 2,     // The handler failed to start.
};

// This is a tri-state of type StartupState. It starts at 0 == kNotReady, and
// when the handler is known to have started successfully, or failed to start
// the value will be updated. The unhandled exception filter will not proceed
// until one of those two cases happens.
base::subtle::AtomicWord g_handler_startup_state;

// A CRITICAL_SECTION initialized with
// RTL_CRITICAL_SECTION_FLAG_FORCE_DEBUG_INFO to force it to be allocated with a
// valid .DebugInfo field. The address of this critical section is given to the
// handler. All critical sections with debug info are linked in a doubly-linked
// list, so this allows the handler to capture all of them.
CRITICAL_SECTION g_critical_section_with_debug_info;

void SetHandlerStartupState(StartupState state) {
  DCHECK(state == StartupState::kSucceeded ||
         state == StartupState::kFailed);
  base::subtle::Acquire_Store(&g_handler_startup_state,
                              static_cast<base::subtle::AtomicWord>(state));
}

StartupState BlockUntilHandlerStartedOrFailed() {
  // Wait until we know the handler has either succeeded or failed to start.
  base::subtle::AtomicWord startup_state;
  while (
      (startup_state = base::subtle::Release_Load(&g_handler_startup_state)) ==
      static_cast<int>(StartupState::kNotReady)) {
    Sleep(1);
  }

  return static_cast<StartupState>(startup_state);
}

#if defined(ADDRESS_SANITIZER)
extern "C" LONG __asan_unhandled_exception_filter(EXCEPTION_POINTERS* info);
#endif

LONG WINAPI UnhandledExceptionHandler(EXCEPTION_POINTERS* exception_pointers) {
#if defined(ADDRESS_SANITIZER)
  // In ASan builds, delegate to the ASan exception filter.
  LONG status = __asan_unhandled_exception_filter(exception_pointers);
  if (status != EXCEPTION_CONTINUE_SEARCH)
    return status;
#endif

  if (BlockUntilHandlerStartedOrFailed() == StartupState::kFailed) {
    // If we know for certain that the handler has failed to start, then abort
    // here, rather than trying to signal to a handler that will never arrive,
    // and then sleeping unnecessarily.
    LOG(ERROR) << "crash server failed to launch, self-terminating";
    SafeTerminateProcess(GetCurrentProcess(), kTerminationCodeCrashNoDump);
    return EXCEPTION_CONTINUE_SEARCH;
  }

  // Otherwise, we know the handler startup has succeeded, and we can continue.

  // Tracks whether a thread has already entered UnhandledExceptionHandler.
  static base::subtle::AtomicWord have_crashed;

  // This is a per-process handler. While this handler is being invoked, other
  // threads are still executing as usual, so multiple threads could enter at
  // the same time. Because we're in a crashing state, we shouldn't be doing
  // anything that might cause allocations, call into kernel mode, etc. So, we
  // don't want to take a critical section here to avoid simultaneous access to
  // the global exception pointers in ExceptionInformation. Because the crash
  // handler will record all threads, it's fine to simply have the second and
  // subsequent entrants block here. They will soon be suspended by the crash
  // handler, and then the entire process will be terminated below. This means
  // that we won't save the exception pointers from the second and further
  // crashes, but contention here is very unlikely, and we'll still have a stack
  // that's blocked at this location.
  if (base::subtle::Barrier_AtomicIncrement(&have_crashed, 1) > 1) {
    SleepEx(INFINITE, false);
  }

  // Otherwise, we're the first thread, so record the exception pointer and
  // signal the crash handler.
  g_crash_exception_information.thread_id = GetCurrentThreadId();
  g_crash_exception_information.exception_pointers =
      FromPointerCast<WinVMAddress>(exception_pointers);

  // Now signal the crash server, which will take a dump and then terminate us
  // when it's complete.
  SetEvent(g_signal_exception);

  // Time to wait for the handler to create a dump.
  constexpr DWORD kMillisecondsUntilTerminate = 60 * 1000;

  // Sleep for a while to allow it to process us. Eventually, we terminate
  // ourselves in case the crash server is gone, so that we don't leave zombies
  // around. This would ideally never happen.
  Sleep(kMillisecondsUntilTerminate);

  LOG(ERROR) << "crash server did not respond, self-terminating";

  SafeTerminateProcess(GetCurrentProcess(), kTerminationCodeCrashNoDump);

  return EXCEPTION_CONTINUE_SEARCH;
}

void HandleAbortSignal(int signum) {
  DCHECK_EQ(signum, SIGABRT);

  CONTEXT context;
  CaptureContext(&context);

  EXCEPTION_RECORD record = {};
  record.ExceptionCode = STATUS_FATAL_APP_EXIT;
  record.ExceptionFlags = EXCEPTION_NONCONTINUABLE;
#if defined(ARCH_CPU_64_BITS)
  record.ExceptionAddress = reinterpret_cast<void*>(context.Rip);
#else
  record.ExceptionAddress = reinterpret_cast<void*>(context.Eip);
#endif  // ARCH_CPU_64_BITS

  EXCEPTION_POINTERS exception_pointers;
  exception_pointers.ContextRecord = &context;
  exception_pointers.ExceptionRecord = &record;

  UnhandledExceptionHandler(&exception_pointers);
}

std::wstring FormatArgumentString(const std::string& name,
                                  const std::wstring& value) {
  return std::wstring(L"--") + base::UTF8ToUTF16(name) + L"=" + value;
}

struct ScopedProcThreadAttributeListTraits {
  static PPROC_THREAD_ATTRIBUTE_LIST InvalidValue() { return nullptr; }

  static void Free(PPROC_THREAD_ATTRIBUTE_LIST proc_thread_attribute_list) {
    // This is able to use GET_FUNCTION_REQUIRED() instead of GET_FUNCTION()
    // because it will only be called if InitializeProcThreadAttributeList() and
    // UpdateProcThreadAttribute() are present.
    static const auto delete_proc_thread_attribute_list =
        GET_FUNCTION_REQUIRED(L"kernel32.dll", ::DeleteProcThreadAttributeList);
    delete_proc_thread_attribute_list(proc_thread_attribute_list);
  }
};

using ScopedProcThreadAttributeList =
    base::ScopedGeneric<PPROC_THREAD_ATTRIBUTE_LIST,
                        ScopedProcThreadAttributeListTraits>;

bool IsInheritableHandle(HANDLE handle) {
  if (!handle || handle == INVALID_HANDLE_VALUE)
    return false;

  // File handles (FILE_TYPE_DISK) and pipe handles (FILE_TYPE_PIPE) are known
  // to be inheritable. Console handles (FILE_TYPE_CHAR) are not inheritable via
  // PROC_THREAD_ATTRIBUTE_HANDLE_LIST. See
  // https://crashpad.chromium.org/bug/77.
  DWORD handle_type = GetFileType(handle);
  return handle_type == FILE_TYPE_DISK || handle_type == FILE_TYPE_PIPE;
}

// Adds |handle| to |handle_list| if it appears valid, and is not already in
// |handle_list|.
//
// Invalid handles (including INVALID_HANDLE_VALUE and null handles) cannot be
// added to a PPROC_THREAD_ATTRIBUTE_LIST’s PROC_THREAD_ATTRIBUTE_HANDLE_LIST.
// If INVALID_HANDLE_VALUE appears, CreateProcess() will fail with
// ERROR_INVALID_PARAMETER. If a null handle appears, the child process will
// silently not inherit any handles.
//
// Use this function to add handles with uncertain validities.
void AddHandleToListIfValidAndInheritable(std::vector<HANDLE>* handle_list,
                                          HANDLE handle) {
  // There doesn't seem to be any documentation of this, but if there's a handle
  // duplicated in this list, CreateProcess() fails with
  // ERROR_INVALID_PARAMETER.
  if (IsInheritableHandle(handle) &&
      std::find(handle_list->begin(), handle_list->end(), handle) ==
          handle_list->end()) {
    handle_list->push_back(handle);
  }
}

void AddUint32(std::vector<unsigned char>* data_vector, uint32_t data) {
  data_vector->push_back(static_cast<unsigned char>(data & 0xff));
  data_vector->push_back(static_cast<unsigned char>((data & 0xff00) >> 8));
  data_vector->push_back(static_cast<unsigned char>((data & 0xff0000) >> 16));
  data_vector->push_back(static_cast<unsigned char>((data & 0xff000000) >> 24));
}

void AddUint64(std::vector<unsigned char>* data_vector, uint64_t data) {
  AddUint32(data_vector, static_cast<uint32_t>(data & 0xffffffffULL));
  AddUint32(data_vector,
            static_cast<uint32_t>((data & 0xffffffff00000000ULL) >> 32));
}

//! \brief Creates a randomized pipe name to listen for client registrations
//!     on and returns its name.
//!
//! \param[out] pipe_name The pipe name that will be listened on.
//! \param[out] pipe_handle The first pipe instance corresponding for the pipe.
void CreatePipe(std::wstring* pipe_name, ScopedFileHANDLE* pipe_instance) {
  int tries = 5;
  std::string pipe_name_base =
      base::StringPrintf("\\\\.\\pipe\\crashpad_%lu_", GetCurrentProcessId());
  do {
    *pipe_name = base::UTF8ToUTF16(pipe_name_base + RandomString());

    pipe_instance->reset(CreateNamedPipeInstance(*pipe_name, true));

    // CreateNamedPipe() is documented as setting the error to
    // ERROR_ACCESS_DENIED if FILE_FLAG_FIRST_PIPE_INSTANCE is specified and the
    // pipe name is already in use. However it may set the error to other codes
    // such as ERROR_PIPE_BUSY (if the pipe already exists and has reached its
    // maximum instance count) or ERROR_INVALID_PARAMETER (if the pipe already
    // exists and its attributes differ from those specified to
    // CreateNamedPipe()). Some of these errors may be ambiguous: for example,
    // ERROR_INVALID_PARAMETER may also occur if CreateNamedPipe() is called
    // incorrectly even in the absence of an existing pipe by the same name.
    // Rather than chasing down all of the possible errors that might indicate
    // that a pipe name is already in use, retry up to a few times on any error.
  } while (!pipe_instance->is_valid() && --tries);

  PCHECK(pipe_instance->is_valid()) << "CreateNamedPipe";
}

struct BackgroundHandlerStartThreadData {
  BackgroundHandlerStartThreadData(
      const base::FilePath& handler,
      const base::FilePath& database,
      const base::FilePath& metrics_dir,
      const std::string& url,
      const std::map<std::string, std::string>& annotations,
      const std::vector<std::string>& arguments,
      const std::wstring& ipc_pipe,
      ScopedFileHANDLE ipc_pipe_handle)
      : handler(handler),
        database(database),
        metrics_dir(metrics_dir),
        url(url),
        annotations(annotations),
        arguments(arguments),
        ipc_pipe(ipc_pipe),
        ipc_pipe_handle(std::move(ipc_pipe_handle)) {}

  base::FilePath handler;
  base::FilePath database;
  base::FilePath metrics_dir;
  std::string url;
  std::map<std::string, std::string> annotations;
  std::vector<std::string> arguments;
  std::wstring ipc_pipe;
  ScopedFileHANDLE ipc_pipe_handle;
};

// Ensures that SetHandlerStartupState() is called on scope exit. Assumes
// failure, and on success, SetSuccessful() should be called.
class ScopedCallSetHandlerStartupState {
 public:
  ScopedCallSetHandlerStartupState() : successful_(false) {}

  ~ScopedCallSetHandlerStartupState() {
    SetHandlerStartupState(successful_ ? StartupState::kSucceeded
                                       : StartupState::kFailed);
  }

  void SetSuccessful() { successful_ = true; }

 private:
  bool successful_;

  DISALLOW_COPY_AND_ASSIGN(ScopedCallSetHandlerStartupState);
};

bool StartHandlerProcess(
    std::unique_ptr<BackgroundHandlerStartThreadData> data) {
  ScopedCallSetHandlerStartupState scoped_startup_state_caller;

  std::wstring command_line;
  AppendCommandLineArgument(data->handler.value(), &command_line);
  for (const std::string& argument : data->arguments) {
    AppendCommandLineArgument(base::UTF8ToUTF16(argument), &command_line);
  }
  if (!data->database.value().empty()) {
    AppendCommandLineArgument(
        FormatArgumentString("database", data->database.value()),
        &command_line);
  }
  if (!data->metrics_dir.value().empty()) {
    AppendCommandLineArgument(
        FormatArgumentString("metrics-dir", data->metrics_dir.value()),
        &command_line);
  }
  if (!data->url.empty()) {
    AppendCommandLineArgument(
        FormatArgumentString("url", base::UTF8ToUTF16(data->url)),
        &command_line);
  }
  for (const auto& kv : data->annotations) {
    AppendCommandLineArgument(
        FormatArgumentString("annotation",
                             base::UTF8ToUTF16(kv.first + '=' + kv.second)),
        &command_line);
  }

  ScopedKernelHANDLE this_process(
      OpenProcess(kXPProcessAllAccess, true, GetCurrentProcessId()));
  if (!this_process.is_valid()) {
    PLOG(ERROR) << "OpenProcess";
    return false;
  }

  InitialClientData initial_client_data(
      g_signal_exception,
      g_signal_non_crash_dump,
      g_non_crash_dump_done,
      data->ipc_pipe_handle.get(),
      this_process.get(),
      FromPointerCast<WinVMAddress>(&g_crash_exception_information),
      FromPointerCast<WinVMAddress>(&g_non_crash_exception_information),
      FromPointerCast<WinVMAddress>(&g_critical_section_with_debug_info));
  AppendCommandLineArgument(
      base::UTF8ToUTF16(std::string("--initial-client-data=") +
                        initial_client_data.StringRepresentation()),
      &command_line);

  BOOL rv;
  DWORD creation_flags;
  STARTUPINFOEX startup_info = {};
  startup_info.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
  startup_info.StartupInfo.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
  startup_info.StartupInfo.hStdOutput = GetStdHandle(STD_OUTPUT_HANDLE);
  startup_info.StartupInfo.hStdError = GetStdHandle(STD_ERROR_HANDLE);

  std::vector<HANDLE> handle_list;
  std::unique_ptr<uint8_t[]> proc_thread_attribute_list_storage;
  ScopedProcThreadAttributeList proc_thread_attribute_list_owner;

  static const auto initialize_proc_thread_attribute_list =
      GET_FUNCTION(L"kernel32.dll", ::InitializeProcThreadAttributeList);
  static const auto update_proc_thread_attribute =
      initialize_proc_thread_attribute_list
          ? GET_FUNCTION(L"kernel32.dll", ::UpdateProcThreadAttribute)
          : nullptr;
  if (!initialize_proc_thread_attribute_list || !update_proc_thread_attribute) {
    // The OS doesn’t allow handle inheritance to be restricted, so the handler
    // will inherit every inheritable handle.
    creation_flags = 0;
    startup_info.StartupInfo.cb = sizeof(startup_info.StartupInfo);
  } else {
    // Restrict handle inheritance to just those needed in the handler.

    creation_flags = EXTENDED_STARTUPINFO_PRESENT;
    startup_info.StartupInfo.cb = sizeof(startup_info);
    SIZE_T size;
    rv = initialize_proc_thread_attribute_list(nullptr, 1, 0, &size);
    if (rv) {
      LOG(ERROR) << "InitializeProcThreadAttributeList (size) succeeded, "
                    "expected failure";
      return false;
    } else if (GetLastError() != ERROR_INSUFFICIENT_BUFFER) {
      PLOG(ERROR) << "InitializeProcThreadAttributeList (size)";
      return false;
    }

    proc_thread_attribute_list_storage.reset(new uint8_t[size]);
    startup_info.lpAttributeList =
        reinterpret_cast<PPROC_THREAD_ATTRIBUTE_LIST>(
            proc_thread_attribute_list_storage.get());
    rv = initialize_proc_thread_attribute_list(
        startup_info.lpAttributeList, 1, 0, &size);
    if (!rv) {
      PLOG(ERROR) << "InitializeProcThreadAttributeList";
      return false;
    }
    proc_thread_attribute_list_owner.reset(startup_info.lpAttributeList);

    handle_list.reserve(8);
    handle_list.push_back(g_signal_exception);
    handle_list.push_back(g_signal_non_crash_dump);
    handle_list.push_back(g_non_crash_dump_done);
    handle_list.push_back(data->ipc_pipe_handle.get());
    handle_list.push_back(this_process.get());
    AddHandleToListIfValidAndInheritable(&handle_list,
                                         startup_info.StartupInfo.hStdInput);
    AddHandleToListIfValidAndInheritable(&handle_list,
                                         startup_info.StartupInfo.hStdOutput);
    AddHandleToListIfValidAndInheritable(&handle_list,
                                         startup_info.StartupInfo.hStdError);
    rv = update_proc_thread_attribute(
        startup_info.lpAttributeList,
        0,
        PROC_THREAD_ATTRIBUTE_HANDLE_LIST,
        &handle_list[0],
        handle_list.size() * sizeof(handle_list[0]),
        nullptr,
        nullptr);
    if (!rv) {
      PLOG(ERROR) << "UpdateProcThreadAttribute";
      return false;
    }
  }

  PROCESS_INFORMATION process_info;
  rv = CreateProcess(data->handler.value().c_str(),
                     &command_line[0],
                     nullptr,
                     nullptr,
                     true,
                     creation_flags,
                     nullptr,
                     nullptr,
                     &startup_info.StartupInfo,
                     &process_info);
  if (!rv) {
    PLOG(ERROR) << "CreateProcess";
    return false;
  }

  rv = CloseHandle(process_info.hThread);
  PLOG_IF(WARNING, !rv) << "CloseHandle thread";

  rv = CloseHandle(process_info.hProcess);
  PLOG_IF(WARNING, !rv) << "CloseHandle process";

  // It is important to close our side of the pipe here before confirming that
  // we can communicate with the server. By doing so, the only remaining copy of
  // the server side of the pipe belongs to the exception handler process we
  // just spawned. Otherwise, the pipe will continue to exist indefinitely, so
  // the connection loop will not detect that it will never be serviced.
  data->ipc_pipe_handle.reset();

  // Confirm that the server is waiting for connections before continuing.
  ClientToServerMessage message = {};
  message.type = ClientToServerMessage::kPing;
  ServerToClientMessage response = {};
  if (!SendToCrashHandlerServer(data->ipc_pipe, message, &response)) {
    return false;
  }

  scoped_startup_state_caller.SetSuccessful();
  return true;
}

DWORD WINAPI BackgroundHandlerStartThreadProc(void* data) {
  std::unique_ptr<BackgroundHandlerStartThreadData> data_as_ptr(
      reinterpret_cast<BackgroundHandlerStartThreadData*>(data));
  return StartHandlerProcess(std::move(data_as_ptr)) ? 0 : 1;
}

void CommonInProcessInitialization() {
  // We create this dummy CRITICAL_SECTION with the
  // RTL_CRITICAL_SECTION_FLAG_FORCE_DEBUG_INFO flag set to have an entry point
  // into the doubly-linked list of RTL_CRITICAL_SECTION_DEBUG objects. This
  // allows us to walk the list at crash time to gather data for !locks. A
  // debugger would instead inspect ntdll!RtlCriticalSectionList to get the head
  // of the list. But that is not an exported symbol, so on an arbitrary client
  // machine, we don't have a way of getting that pointer.
  InitializeCriticalSectionWithDebugInfoIfPossible(
      &g_critical_section_with_debug_info);

  g_non_crash_dump_lock = new base::Lock();
}

void RegisterHandlers() {
  SetUnhandledExceptionFilter(&UnhandledExceptionHandler);

  // The Windows CRT's signal.h lists:
  // - SIGINT
  // - SIGILL
  // - SIGFPE
  // - SIGSEGV
  // - SIGTERM
  // - SIGBREAK
  // - SIGABRT
  // SIGILL and SIGTERM are documented as not being generated. SIGBREAK and
  // SIGINT are for Ctrl-Break and Ctrl-C, and aren't something for which
  // capturing a dump is warranted. SIGFPE and SIGSEGV are captured as regular
  // exceptions through the unhandled exception filter. This leaves SIGABRT. In
  // the standard CRT, abort() is implemented as a synchronous call to the
  // SIGABRT signal handler if installed, but after doing so, the unhandled
  // exception filter is not triggered (it instead __fastfail()s). So, register
  // to handle SIGABRT to catch abort() calls, as client code might use this and
  // expect it to cause a crash dump. This will only work when the abort()
  // that's called in client code is the same (or has the same behavior) as the
  // one in use here.
  void (*rv)(int) = signal(SIGABRT, HandleAbortSignal);
  DCHECK_NE(rv, SIG_ERR);
}

}  // namespace

CrashpadClient::CrashpadClient() : ipc_pipe_(), handler_start_thread_() {}

CrashpadClient::~CrashpadClient() {}

bool CrashpadClient::StartHandler(
    const base::FilePath& handler,
    const base::FilePath& database,
    const base::FilePath& metrics_dir,
    const std::string& url,
    const std::map<std::string, std::string>& annotations,
    const std::vector<std::string>& arguments,
    bool restartable,
    bool asynchronous_start) {
  DCHECK(ipc_pipe_.empty());

  // Both the pipe and the signalling events have to be created on the main
  // thread (not the spawning thread) so that they're valid after we return from
  // this function.
  ScopedFileHANDLE ipc_pipe_handle;
  CreatePipe(&ipc_pipe_, &ipc_pipe_handle);

  SECURITY_ATTRIBUTES security_attributes = {0};
  security_attributes.nLength = sizeof(SECURITY_ATTRIBUTES);
  security_attributes.bInheritHandle = true;

  g_signal_exception =
      CreateEvent(&security_attributes, false /* auto reset */, false, nullptr);
  g_signal_non_crash_dump =
      CreateEvent(&security_attributes, false /* auto reset */, false, nullptr);
  g_non_crash_dump_done =
      CreateEvent(&security_attributes, false /* auto reset */, false, nullptr);

  CommonInProcessInitialization();

  RegisterHandlers();

  auto data = new BackgroundHandlerStartThreadData(handler,
                                                   database,
                                                   metrics_dir,
                                                   url,
                                                   annotations,
                                                   arguments,
                                                   ipc_pipe_,
                                                   std::move(ipc_pipe_handle));

  if (asynchronous_start) {
    // It is important that the current thread not be synchronized with the
    // thread that is created here. StartHandler() needs to be callable inside a
    // DllMain(). In that case, the background thread will not start until the
    // current DllMain() completes, which would cause deadlock if it was waited
    // upon.
    handler_start_thread_.reset(CreateThread(nullptr,
                                             0,
                                             &BackgroundHandlerStartThreadProc,
                                             reinterpret_cast<void*>(data),
                                             0,
                                             nullptr));
    if (!handler_start_thread_.is_valid()) {
      PLOG(ERROR) << "CreateThread";
      SetHandlerStartupState(StartupState::kFailed);
      return false;
    }

    // In asynchronous mode, we can't report on the overall success or failure
    // of initialization at this point.
    return true;
  } else {
    return StartHandlerProcess(
        std::unique_ptr<BackgroundHandlerStartThreadData>(data));
  }
}

bool CrashpadClient::SetHandlerIPCPipe(const std::wstring& ipc_pipe) {
  DCHECK(ipc_pipe_.empty());
  DCHECK(!ipc_pipe.empty());

  ipc_pipe_ = ipc_pipe;

  DCHECK(!ipc_pipe_.empty());
  DCHECK_EQ(g_signal_exception, INVALID_HANDLE_VALUE);
  DCHECK_EQ(g_signal_non_crash_dump, INVALID_HANDLE_VALUE);
  DCHECK_EQ(g_non_crash_dump_done, INVALID_HANDLE_VALUE);
  DCHECK(!g_critical_section_with_debug_info.DebugInfo);
  DCHECK(!g_non_crash_dump_lock);

  ClientToServerMessage message;
  memset(&message, 0, sizeof(message));
  message.type = ClientToServerMessage::kRegister;
  message.registration.version = RegistrationRequest::kMessageVersion;
  message.registration.client_process_id = GetCurrentProcessId();
  message.registration.crash_exception_information =
      FromPointerCast<WinVMAddress>(&g_crash_exception_information);
  message.registration.non_crash_exception_information =
      FromPointerCast<WinVMAddress>(&g_non_crash_exception_information);

  CommonInProcessInitialization();

  message.registration.critical_section_address =
      FromPointerCast<WinVMAddress>(&g_critical_section_with_debug_info);

  ServerToClientMessage response = {};

  if (!SendToCrashHandlerServer(ipc_pipe_, message, &response)) {
    return false;
  }

  SetHandlerStartupState(StartupState::kSucceeded);

  RegisterHandlers();

  // The server returns these already duplicated to be valid in this process.
  g_signal_exception =
      IntToHandle(response.registration.request_crash_dump_event);
  g_signal_non_crash_dump =
      IntToHandle(response.registration.request_non_crash_dump_event);
  g_non_crash_dump_done =
      IntToHandle(response.registration.non_crash_dump_completed_event);

  return true;
}

std::wstring CrashpadClient::GetHandlerIPCPipe() const {
  DCHECK(!ipc_pipe_.empty());
  return ipc_pipe_;
}

bool CrashpadClient::WaitForHandlerStart(unsigned int timeout_ms) {
  DCHECK(handler_start_thread_.is_valid());
  DWORD result = WaitForSingleObject(handler_start_thread_.get(), timeout_ms);
  if (result == WAIT_TIMEOUT) {
    LOG(ERROR) << "WaitForSingleObject timed out";
    return false;
  } else if (result == WAIT_ABANDONED) {
    LOG(ERROR) << "WaitForSingleObject abandoned";
    return false;
  } else if (result != WAIT_OBJECT_0) {
    PLOG(ERROR) << "WaitForSingleObject";
    return false;
  }

  DWORD exit_code;
  if (!GetExitCodeThread(handler_start_thread_.get(), &exit_code)) {
    PLOG(ERROR) << "GetExitCodeThread";
    return false;
  }

  handler_start_thread_.reset();
  return exit_code == 0;
}

// static
void CrashpadClient::DumpWithoutCrash(const CONTEXT& context) {
  if (g_signal_non_crash_dump == INVALID_HANDLE_VALUE ||
      g_non_crash_dump_done == INVALID_HANDLE_VALUE) {
    LOG(ERROR) << "not connected";
    return;
  }

  if (BlockUntilHandlerStartedOrFailed() == StartupState::kFailed) {
    // If we know for certain that the handler has failed to start, then abort
    // here, as we would otherwise wait indefinitely for the
    // g_non_crash_dump_done event that would never be signalled.
    LOG(ERROR) << "crash server failed to launch, no dump captured";
    return;
  }

  // In the non-crashing case, we aren't concerned about avoiding calls into
  // Win32 APIs, so just use regular locking here in case of multiple threads
  // calling this function. If a crash occurs while we're in here, the worst
  // that can happen is that the server captures a partial dump for this path
  // because another thread’s crash processing finished and the process was
  // terminated before this thread’s non-crash processing could be completed.
  base::AutoLock lock(*g_non_crash_dump_lock);

  // Create a fake EXCEPTION_POINTERS to give the handler something to work
  // with.
  EXCEPTION_POINTERS exception_pointers = {};

  // This is logically const, but EXCEPTION_POINTERS does not declare it as
  // const, so we have to cast that away from the argument.
  exception_pointers.ContextRecord = const_cast<CONTEXT*>(&context);

  // We include a fake exception and use a code of '0x517a7ed' (something like
  // "simulated") so that it's relatively obvious in windbg that it's not
  // actually an exception. Most values in
  // https://msdn.microsoft.com/library/aa363082.aspx have some of the top
  // nibble set, so we make sure to pick a value that doesn't, so as to be
  // unlikely to conflict.
  constexpr uint32_t kSimulatedExceptionCode = 0x517a7ed;
  EXCEPTION_RECORD record = {};
  record.ExceptionCode = kSimulatedExceptionCode;
#if defined(ARCH_CPU_64_BITS)
  record.ExceptionAddress = reinterpret_cast<void*>(context.Rip);
#else
  record.ExceptionAddress = reinterpret_cast<void*>(context.Eip);
#endif  // ARCH_CPU_64_BITS

  exception_pointers.ExceptionRecord = &record;

  g_non_crash_exception_information.thread_id = GetCurrentThreadId();
  g_non_crash_exception_information.exception_pointers =
      FromPointerCast<WinVMAddress>(&exception_pointers);

  bool set_event_result = !!SetEvent(g_signal_non_crash_dump);
  PLOG_IF(ERROR, !set_event_result) << "SetEvent";

  DWORD wfso_result = WaitForSingleObject(g_non_crash_dump_done, INFINITE);
  PLOG_IF(ERROR, wfso_result != WAIT_OBJECT_0) << "WaitForSingleObject";
}

// static
void CrashpadClient::DumpAndCrash(EXCEPTION_POINTERS* exception_pointers) {
  if (g_signal_exception == INVALID_HANDLE_VALUE) {
    LOG(ERROR) << "not connected";
    SafeTerminateProcess(GetCurrentProcess(),
                         kTerminationCodeNotConnectedToHandler);
    return;
  }

  // We don't need to check for handler startup here, as
  // UnhandledExceptionHandler() necessarily does that.

  UnhandledExceptionHandler(exception_pointers);
}

// static
bool CrashpadClient::DumpAndCrashTargetProcess(HANDLE process,
                                               HANDLE blame_thread,
                                               DWORD exception_code) {
  // Confirm we're on Vista or later.
  const DWORD version = GetVersion();
  const DWORD major_version = LOBYTE(LOWORD(version));
  if (major_version < 6) {
    LOG(ERROR) << "unavailable before Vista";
    return false;
  }

  // Confirm that our bitness is the same as the process we're crashing.
  ProcessInfo process_info;
  if (!process_info.Initialize(process)) {
    LOG(ERROR) << "ProcessInfo::Initialize";
    return false;
  }
#if defined(ARCH_CPU_64_BITS)
  if (!process_info.Is64Bit()) {
    LOG(ERROR) << "DumpAndCrashTargetProcess currently not supported x64->x86";
    return false;
  }
#endif  // ARCH_CPU_64_BITS

  ScopedProcessSuspend suspend(process);

  // If no thread handle was provided, or the thread has already exited, we pass
  // 0 to the handler, which indicates no fake exception record to be created.
  DWORD thread_id = 0;
  if (blame_thread) {
    // Now that we've suspended the process, if our thread hasn't exited, we
    // know we're relatively safe to pass the thread id through.
    if (WaitForSingleObject(blame_thread, 0) == WAIT_TIMEOUT) {
      static const auto get_thread_id =
          GET_FUNCTION_REQUIRED(L"kernel32.dll", ::GetThreadId);
      thread_id = get_thread_id(blame_thread);
    }
  }

  constexpr size_t kInjectBufferSize = 4 * 1024;
  WinVMAddress inject_memory =
      FromPointerCast<WinVMAddress>(VirtualAllocEx(process,
                                                   nullptr,
                                                   kInjectBufferSize,
                                                   MEM_RESERVE | MEM_COMMIT,
                                                   PAGE_READWRITE));
  if (!inject_memory) {
    PLOG(ERROR) << "VirtualAllocEx";
    return false;
  }

  // Because we're the same bitness as our target, we can rely kernel32 being
  // loaded at the same address in our process as the target, and just look up
  // its address here.
  WinVMAddress raise_exception_address =
      FromPointerCast<WinVMAddress>(&RaiseException);

  WinVMAddress code_entry_point = 0;
  std::vector<unsigned char> data_to_write;
  if (process_info.Is64Bit()) {
    // Data written is first, the data for the 4th argument (lpArguments) to
    // RaiseException(). A two element array:
    //
    // DWORD64: thread_id
    // DWORD64: exception_code
    //
    // Following that, code which sets the arguments to RaiseException() and
    // then calls it:
    //
    // mov r9, <data_array_address>
    // mov r8d, 2  ; nNumberOfArguments
    // mov edx, 1  ; dwExceptionFlags = EXCEPTION_NONCONTINUABLE
    // mov ecx, 0xcca11ed  ; dwExceptionCode, interpreted specially by the
    //                     ;                  handler.
    // jmp <address_of_RaiseException>
    //
    // Note that the first three arguments to RaiseException() are DWORDs even
    // on x64, so only the 4th argument (a pointer) is a full-width register.
    //
    // We also don't need to set up a stack or use call, since the only
    // registers modified are volatile ones, and we can just jmp straight to
    // RaiseException().

    // The data array.
    AddUint64(&data_to_write, thread_id);
    AddUint64(&data_to_write, exception_code);

    // The thread entry point.
    code_entry_point = inject_memory + data_to_write.size();

    // r9 = pointer to data.
    data_to_write.push_back(0x49);
    data_to_write.push_back(0xb9);
    AddUint64(&data_to_write, inject_memory);

    // r8d = 2 for nNumberOfArguments.
    data_to_write.push_back(0x41);
    data_to_write.push_back(0xb8);
    AddUint32(&data_to_write, 2);

    // edx = 1 for dwExceptionFlags.
    data_to_write.push_back(0xba);
    AddUint32(&data_to_write, 1);

    // ecx = kTriggeredExceptionCode for dwExceptionCode.
    data_to_write.push_back(0xb9);
    AddUint32(&data_to_write, kTriggeredExceptionCode);

    // jmp to RaiseException() via rax.
    data_to_write.push_back(0x48);  // mov rax, imm.
    data_to_write.push_back(0xb8);
    AddUint64(&data_to_write, raise_exception_address);
    data_to_write.push_back(0xff);  // jmp rax.
    data_to_write.push_back(0xe0);
  } else {
    // Data written is first, the data for the 4th argument (lpArguments) to
    // RaiseException(). A two element array:
    //
    // DWORD: thread_id
    // DWORD: exception_code
    //
    // Following that, code which pushes our arguments to RaiseException() and
    // then calls it:
    //
    // push <data_array_address>
    // push 2  ; nNumberOfArguments
    // push 1  ; dwExceptionFlags = EXCEPTION_NONCONTINUABLE
    // push 0xcca11ed  ; dwExceptionCode, interpreted specially by the handler.
    // call <address_of_RaiseException>
    // ud2  ; Generate invalid opcode to make sure we still crash if we return
    //      ; for some reason.
    //
    // No need to clean up the stack, as RaiseException() is __stdcall.

    // The data array.
    AddUint32(&data_to_write, thread_id);
    AddUint32(&data_to_write, exception_code);

    // The thread entry point.
    code_entry_point = inject_memory + data_to_write.size();

    // Push data address.
    data_to_write.push_back(0x68);
    AddUint32(&data_to_write, static_cast<uint32_t>(inject_memory));

    // Push 2 for nNumberOfArguments.
    data_to_write.push_back(0x6a);
    data_to_write.push_back(2);

    // Push 1 for dwExceptionCode.
    data_to_write.push_back(0x6a);
    data_to_write.push_back(1);

    // Push dwExceptionFlags.
    data_to_write.push_back(0x68);
    AddUint32(&data_to_write, kTriggeredExceptionCode);

    // Relative call to RaiseException().
    int64_t relative_address_to_raise_exception =
        raise_exception_address - (inject_memory + data_to_write.size() + 5);
    data_to_write.push_back(0xe8);
    AddUint32(&data_to_write,
              static_cast<uint32_t>(relative_address_to_raise_exception));

    // ud2.
    data_to_write.push_back(0x0f);
    data_to_write.push_back(0x0b);
  }

  DCHECK_LT(data_to_write.size(), kInjectBufferSize);

  SIZE_T bytes_written;
  if (!WriteProcessMemory(process,
                          reinterpret_cast<void*>(inject_memory),
                          data_to_write.data(),
                          data_to_write.size(),
                          &bytes_written)) {
    PLOG(ERROR) << "WriteProcessMemory";
    return false;
  }

  if (bytes_written != data_to_write.size()) {
    LOG(ERROR) << "WriteProcessMemory unexpected number of bytes";
    return false;
  }

  if (!FlushInstructionCache(
          process, reinterpret_cast<void*>(inject_memory), bytes_written)) {
    PLOG(ERROR) << "FlushInstructionCache";
    return false;
  }

  DWORD old_protect;
  if (!VirtualProtectEx(process,
                        reinterpret_cast<void*>(inject_memory),
                        kInjectBufferSize,
                        PAGE_EXECUTE_READ,
                        &old_protect)) {
    PLOG(ERROR) << "VirtualProtectEx";
    return false;
  }

  // Cause an exception in the target process by creating a thread which calls
  // RaiseException with our arguments above. Note that we cannot get away with
  // using DebugBreakProcess() (nothing happens unless a debugger is attached)
  // and we cannot get away with CreateRemoteThread() because it doesn't work if
  // the target is hung waiting for the loader lock. We use NtCreateThreadEx()
  // with the SKIP_THREAD_ATTACH flag, which skips various notifications,
  // letting this cause an exception, even when the target is stuck in the
  // loader lock.
  HANDLE injected_thread;

  // This is what DebugBreakProcess() uses.
  constexpr size_t kStackSize = 0x4000;

  NTSTATUS status = NtCreateThreadEx(&injected_thread,
                                     STANDARD_RIGHTS_ALL | SPECIFIC_RIGHTS_ALL,
                                     nullptr,
                                     process,
                                     reinterpret_cast<void*>(code_entry_point),
                                     nullptr,
                                     THREAD_CREATE_FLAGS_SKIP_THREAD_ATTACH,
                                     0,
                                     kStackSize,
                                     0,
                                     nullptr);
  if (!NT_SUCCESS(status)) {
    NTSTATUS_LOG(ERROR, status) << "NtCreateThreadEx";
    return false;
  }

  // The injected thread raises an exception and ultimately results in process
  // termination. The suspension must be made aware that the process may be
  // terminating, otherwise it’ll log an extraneous error.
  suspend.TolerateTermination();

  bool result = true;
  if (WaitForSingleObject(injected_thread, 60 * 1000) != WAIT_OBJECT_0) {
    PLOG(ERROR) << "WaitForSingleObject";
    result = false;
  }

  status = NtClose(injected_thread);
  if (!NT_SUCCESS(status)) {
    NTSTATUS_LOG(ERROR, status) << "NtClose";
    result = false;
  }

  return result;
}

}  // namespace crashpad
