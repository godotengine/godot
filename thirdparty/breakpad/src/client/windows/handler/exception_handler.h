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

// ExceptionHandler can write a minidump file when an exception occurs,
// or when WriteMinidump() is called explicitly by your program.
//
// To have the exception handler write minidumps when an uncaught exception
// (crash) occurs, you should create an instance early in the execution
// of your program, and keep it around for the entire time you want to
// have crash handling active (typically, until shutdown).
//
// If you want to write minidumps without installing the exception handler,
// you can create an ExceptionHandler with install_handler set to false,
// then call WriteMinidump.  You can also use this technique if you want to
// use different minidump callbacks for different call sites.
//
// In either case, a callback function is called when a minidump is written,
// which receives the unqiue id of the minidump.  The caller can use this
// id to collect and write additional application state, and to launch an
// external crash-reporting application.
//
// It is important that creation and destruction of ExceptionHandler objects
// be nested cleanly, when using install_handler = true.
// Avoid the following pattern:
//   ExceptionHandler *e = new ExceptionHandler(...);
//   ExceptionHandler *f = new ExceptionHandler(...);
//   delete e;
// This will put the exception filter stack into an inconsistent state.

#ifndef CLIENT_WINDOWS_HANDLER_EXCEPTION_HANDLER_H__
#define CLIENT_WINDOWS_HANDLER_EXCEPTION_HANDLER_H__

#include <stdlib.h>
#include <windows.h>
#include <dbghelp.h>
#include <rpc.h>

#pragma warning(push)
// Disable exception handler warnings.
#pragma warning(disable:4530)

#include <list>
#include <string>
#include <vector>

#include "client/windows/common/ipc_protocol.h"
#include "client/windows/crash_generation/crash_generation_client.h"
#include "common/scoped_ptr.h"
#include "google_breakpad/common/minidump_format.h"

namespace google_breakpad {

using std::vector;
using std::wstring;

// These entries store a list of memory regions that the client wants included
// in the minidump.
struct AppMemory {
  ULONG64 ptr;
  ULONG length;

  bool operator==(const struct AppMemory& other) const {
    return ptr == other.ptr;
  }

  bool operator==(const void* other) const {
    return ptr == reinterpret_cast<ULONG64>(other);
  }
};
typedef std::list<AppMemory> AppMemoryList;

class ExceptionHandler {
 public:
  // A callback function to run before Breakpad performs any substantial
  // processing of an exception.  A FilterCallback is called before writing
  // a minidump.  context is the parameter supplied by the user as
  // callback_context when the handler was created.  exinfo points to the
  // exception record, if any; assertion points to assertion information,
  // if any.
  //
  // If a FilterCallback returns true, Breakpad will continue processing,
  // attempting to write a minidump.  If a FilterCallback returns false,
  // Breakpad will immediately report the exception as unhandled without
  // writing a minidump, allowing another handler the opportunity to handle it.
  typedef bool (*FilterCallback)(void* context, EXCEPTION_POINTERS* exinfo,
                                 MDRawAssertionInfo* assertion);

  // A callback function to run after the minidump has been written.
  // minidump_id is a unique id for the dump, so the minidump
  // file is <dump_path>\<minidump_id>.dmp.  context is the parameter supplied
  // by the user as callback_context when the handler was created.  exinfo
  // points to the exception record, or NULL if no exception occurred.
  // succeeded indicates whether a minidump file was successfully written.
  // assertion points to information about an assertion if the handler was
  // invoked by an assertion.
  //
  // If an exception occurred and the callback returns true, Breakpad will treat
  // the exception as fully-handled, suppressing any other handlers from being
  // notified of the exception.  If the callback returns false, Breakpad will
  // treat the exception as unhandled, and allow another handler to handle it.
  // If there are no other handlers, Breakpad will report the exception to the
  // system as unhandled, allowing a debugger or native crash dialog the
  // opportunity to handle the exception.  Most callback implementations
  // should normally return the value of |succeeded|, or when they wish to
  // not report an exception of handled, false.  Callbacks will rarely want to
  // return true directly (unless |succeeded| is true).
  //
  // For out-of-process dump generation, dump path and minidump ID will always
  // be NULL. In case of out-of-process dump generation, the dump path and
  // minidump id are controlled by the server process and are not communicated
  // back to the crashing process.
  typedef bool (*MinidumpCallback)(const wchar_t* dump_path,
                                   const wchar_t* minidump_id,
                                   void* context,
                                   EXCEPTION_POINTERS* exinfo,
                                   MDRawAssertionInfo* assertion,
                                   bool succeeded);

  // HandlerType specifies which types of handlers should be installed, if
  // any.  Use HANDLER_NONE for an ExceptionHandler that remains idle,
  // without catching any failures on its own.  This type of handler may
  // still be triggered by calling WriteMinidump.  Otherwise, use a
  // combination of the other HANDLER_ values, or HANDLER_ALL to install
  // all handlers.
  enum HandlerType {
    HANDLER_NONE = 0,
    HANDLER_EXCEPTION = 1 << 0,          // SetUnhandledExceptionFilter
    HANDLER_INVALID_PARAMETER = 1 << 1,  // _set_invalid_parameter_handler
    HANDLER_PURECALL = 1 << 2,           // _set_purecall_handler
    HANDLER_ALL = HANDLER_EXCEPTION |
                  HANDLER_INVALID_PARAMETER |
                  HANDLER_PURECALL
  };

  // Creates a new ExceptionHandler instance to handle writing minidumps.
  // Before writing a minidump, the optional filter callback will be called.
  // Its return value determines whether or not Breakpad should write a
  // minidump.  Minidump files will be written to dump_path, and the optional
  // callback is called after writing the dump file, as described above.
  // handler_types specifies the types of handlers that should be installed.
  ExceptionHandler(const wstring& dump_path,
                   FilterCallback filter,
                   MinidumpCallback callback,
                   void* callback_context,
                   int handler_types);

  // Creates a new ExceptionHandler instance that can attempt to perform
  // out-of-process dump generation if pipe_name is not NULL. If pipe_name is
  // NULL, or if out-of-process dump generation registration step fails,
  // in-process dump generation will be used. This also allows specifying
  // the dump type to generate.
  ExceptionHandler(const wstring& dump_path,
                   FilterCallback filter,
                   MinidumpCallback callback,
                   void* callback_context,
                   int handler_types,
                   MINIDUMP_TYPE dump_type,
                   const wchar_t* pipe_name,
                   const CustomClientInfo* custom_info);

  // As above, creates a new ExceptionHandler instance to perform
  // out-of-process dump generation if the given pipe_handle is not NULL.
  ExceptionHandler(const wstring& dump_path,
                   FilterCallback filter,
                   MinidumpCallback callback,
                   void* callback_context,
                   int handler_types,
                   MINIDUMP_TYPE dump_type,
                   HANDLE pipe_handle,
                   const CustomClientInfo* custom_info);

  // ExceptionHandler that ENSURES out-of-process dump generation.  Expects a
  // crash generation client that is already registered with a crash generation
  // server.  Takes ownership of the passed-in crash_generation_client.
  //
  // Usage example:
  //   crash_generation_client = new CrashGenerationClient(..);
  //   if (crash_generation_client->Register()) {
  //     // Registration with the crash generation server succeeded.
  //     // Out-of-process dump generation is guaranteed.
  //     g_handler = new ExceptionHandler(.., crash_generation_client, ..);
  //     return true;
  //   }
  ExceptionHandler(const wstring& dump_path,
                   FilterCallback filter,
                   MinidumpCallback callback,
                   void* callback_context,
                   int handler_types,
                   CrashGenerationClient* crash_generation_client);

  ~ExceptionHandler();

  // Get and set the minidump path.
  wstring dump_path() const { return dump_path_; }
  void set_dump_path(const wstring& dump_path) {
    dump_path_ = dump_path;
    dump_path_c_ = dump_path_.c_str();
    UpdateNextID();  // Necessary to put dump_path_ in next_minidump_path_.
  }

  // Requests that a previously reported crash be uploaded.
  bool RequestUpload(DWORD crash_id);

  // Writes a minidump immediately.  This can be used to capture the
  // execution state independently of a crash.  Returns true on success.
  bool WriteMinidump();

  // Writes a minidump immediately, with the user-supplied exception
  // information.
  bool WriteMinidumpForException(EXCEPTION_POINTERS* exinfo);

  // Convenience form of WriteMinidump which does not require an
  // ExceptionHandler instance.
  static bool WriteMinidump(const wstring& dump_path,
                            MinidumpCallback callback, void* callback_context,
                            MINIDUMP_TYPE dump_type = MiniDumpNormal);

  // Write a minidump of |child| immediately.  This can be used to
  // capture the execution state of |child| independently of a crash.
  // Pass a meaningful |child_blamed_thread| to make that thread in
  // the child process the one from which a crash signature is
  // extracted.
  static bool WriteMinidumpForChild(HANDLE child,
                                    DWORD child_blamed_thread,
                                    const wstring& dump_path,
                                    MinidumpCallback callback,
                                    void* callback_context,
                                    MINIDUMP_TYPE dump_type = MiniDumpNormal);

  // Get the thread ID of the thread requesting the dump (either the exception
  // thread or any other thread that called WriteMinidump directly).  This
  // may be useful if you want to include additional thread state in your
  // dumps.
  DWORD get_requesting_thread_id() const { return requesting_thread_id_; }

  // Controls behavior of EXCEPTION_BREAKPOINT and EXCEPTION_SINGLE_STEP.
  bool get_handle_debug_exceptions() const { return handle_debug_exceptions_; }
  void set_handle_debug_exceptions(bool handle_debug_exceptions) {
    handle_debug_exceptions_ = handle_debug_exceptions;
  }

  // Controls behavior of EXCEPTION_INVALID_HANDLE.
  bool get_consume_invalid_handle_exceptions() const {
    return consume_invalid_handle_exceptions_;
  }
  void set_consume_invalid_handle_exceptions(
      bool consume_invalid_handle_exceptions) {
    consume_invalid_handle_exceptions_ = consume_invalid_handle_exceptions;
  }

  // Returns whether out-of-process dump generation is used or not.
  bool IsOutOfProcess() const { return crash_generation_client_.get() != NULL; }

  // Calling RegisterAppMemory(p, len) causes len bytes starting
  // at address p to be copied to the minidump when a crash happens.
  void RegisterAppMemory(void* ptr, size_t length);
  void UnregisterAppMemory(void* ptr);

 private:
  friend class AutoExceptionHandler;

  // Initializes the instance with given values.
  void Initialize(const wstring& dump_path,
                  FilterCallback filter,
                  MinidumpCallback callback,
                  void* callback_context,
                  int handler_types,
                  MINIDUMP_TYPE dump_type,
                  const wchar_t* pipe_name,
                  HANDLE pipe_handle,
                  CrashGenerationClient* crash_generation_client,
                  const CustomClientInfo* custom_info);

  // Function pointer type for MiniDumpWriteDump, which is looked up
  // dynamically.
  typedef BOOL (WINAPI *MiniDumpWriteDump_type)(
      HANDLE hProcess,
      DWORD dwPid,
      HANDLE hFile,
      MINIDUMP_TYPE DumpType,
      CONST PMINIDUMP_EXCEPTION_INFORMATION ExceptionParam,
      CONST PMINIDUMP_USER_STREAM_INFORMATION UserStreamParam,
      CONST PMINIDUMP_CALLBACK_INFORMATION CallbackParam);

  // Function pointer type for UuidCreate, which is looked up dynamically.
  typedef RPC_STATUS (RPC_ENTRY *UuidCreate_type)(UUID* Uuid);

  // Runs the main loop for the exception handler thread.
  static DWORD WINAPI ExceptionHandlerThreadMain(void* lpParameter);

  // Called on the exception thread when an unhandled exception occurs.
  // Signals the exception handler thread to handle the exception.
  static LONG WINAPI HandleException(EXCEPTION_POINTERS* exinfo);

#if _MSC_VER >= 1400  // MSVC 2005/8
  // This function will be called by some CRT functions when they detect
  // that they were passed an invalid parameter.  Note that in _DEBUG builds,
  // the CRT may display an assertion dialog before calling this function,
  // and the function will not be called unless the assertion dialog is
  // dismissed by clicking "Ignore."
  static void HandleInvalidParameter(const wchar_t* expression,
                                     const wchar_t* function,
                                     const wchar_t* file,
                                     unsigned int line,
                                     uintptr_t reserved);
#endif  // _MSC_VER >= 1400

  // This function will be called by the CRT when a pure virtual
  // function is called.
  static void HandlePureVirtualCall();

  // This is called on the exception thread or on another thread that
  // the user wishes to produce a dump from.  It calls
  // WriteMinidumpWithException on the handler thread, avoiding stack
  // overflows and inconsistent dumps due to writing the dump from
  // the exception thread.  If the dump is requested as a result of an
  // exception, exinfo contains exception information, otherwise, it
  // is NULL.  If the dump is requested as a result of an assertion
  // (such as an invalid parameter being passed to a CRT function),
  // assertion contains data about the assertion, otherwise, it is NULL.
  bool WriteMinidumpOnHandlerThread(EXCEPTION_POINTERS* exinfo,
                                    MDRawAssertionInfo* assertion);

  // This function is called on the handler thread.  It calls into
  // WriteMinidumpWithExceptionForProcess() with a handle to the
  // current process.  requesting_thread_id is the ID of the thread
  // that requested the dump.  If the dump is requested as a result of
  // an exception, exinfo contains exception information, otherwise,
  // it is NULL.
  bool WriteMinidumpWithException(DWORD requesting_thread_id,
                                  EXCEPTION_POINTERS* exinfo,
                                  MDRawAssertionInfo* assertion);

  // This function is used as a callback when calling MinidumpWriteDump,
  // in order to add additional memory regions to the dump.
  static BOOL CALLBACK MinidumpWriteDumpCallback(
      PVOID context,
      const PMINIDUMP_CALLBACK_INPUT callback_input,
      PMINIDUMP_CALLBACK_OUTPUT callback_output);

  // This function does the actual writing of a minidump.  It is
  // called on the handler thread.  requesting_thread_id is the ID of
  // the thread that requested the dump, if that information is
  // meaningful.  If the dump is requested as a result of an
  // exception, exinfo contains exception information, otherwise, it
  // is NULL.  process is the one that will be dumped.  If
  // requesting_thread_id is meaningful and should be added to the
  // minidump, write_requester_stream is |true|.
  bool WriteMinidumpWithExceptionForProcess(DWORD requesting_thread_id,
                                            EXCEPTION_POINTERS* exinfo,
                                            MDRawAssertionInfo* assertion,
                                            HANDLE process,
                                            bool write_requester_stream);

  // Generates a new ID and stores it in next_minidump_id_, and stores the
  // path of the next minidump to be written in next_minidump_path_.
  void UpdateNextID();

  FilterCallback filter_;
  MinidumpCallback callback_;
  void* callback_context_;

  scoped_ptr<CrashGenerationClient> crash_generation_client_;

  // The directory in which a minidump will be written, set by the dump_path
  // argument to the constructor, or set_dump_path.
  wstring dump_path_;

  // The basename of the next minidump to be written, without the extension.
  wstring next_minidump_id_;

  // The full pathname of the next minidump to be written, including the file
  // extension.
  wstring next_minidump_path_;

  // Pointers to C-string representations of the above.  These are set when
  // the above wstring versions are set in order to avoid calling c_str during
  // an exception, as c_str may attempt to allocate heap memory.  These
  // pointers are not owned by the ExceptionHandler object, but their lifetimes
  // should be equivalent to the lifetimes of the associated wstring, provided
  // that the wstrings are not altered.
  const wchar_t* dump_path_c_;
  const wchar_t* next_minidump_id_c_;
  const wchar_t* next_minidump_path_c_;

  HMODULE dbghelp_module_;
  MiniDumpWriteDump_type minidump_write_dump_;
  MINIDUMP_TYPE dump_type_;

  HMODULE rpcrt4_module_;
  UuidCreate_type uuid_create_;

  // Tracks the handler types that were installed according to the
  // handler_types constructor argument.
  int handler_types_;

  // When installed_handler_ is true, previous_filter_ is the unhandled
  // exception filter that was set prior to installing ExceptionHandler as
  // the unhandled exception filter and pointing it to |this|.  NULL indicates
  // that there is no previous unhandled exception filter.
  LPTOP_LEVEL_EXCEPTION_FILTER previous_filter_;

#if _MSC_VER >= 1400  // MSVC 2005/8
  // Beginning in VC 8, the CRT provides an invalid parameter handler that will
  // be called when some CRT functions are passed invalid parameters.  In
  // earlier CRTs, the same conditions would cause unexpected behavior or
  // crashes.
  _invalid_parameter_handler previous_iph_;
#endif  // _MSC_VER >= 1400

  // The CRT allows you to override the default handler for pure
  // virtual function calls.
  _purecall_handler previous_pch_;

  // The exception handler thread.
  HANDLE handler_thread_;

  // True if the exception handler is being destroyed.
  // Starting with MSVC 2005, Visual C has stronger guarantees on volatile vars.
  // It has release semantics on write and acquire semantics on reads.
  // See the msdn documentation.
  volatile bool is_shutdown_;

  // The critical section enforcing the requirement that only one exception be
  // handled by a handler at a time.
  CRITICAL_SECTION handler_critical_section_;

  // Semaphores used to move exception handling between the exception thread
  // and the handler thread.  handler_start_semaphore_ is signalled by the
  // exception thread to wake up the handler thread when an exception occurs.
  // handler_finish_semaphore_ is signalled by the handler thread to wake up
  // the exception thread when handling is complete.
  HANDLE handler_start_semaphore_;
  HANDLE handler_finish_semaphore_;

  // The next 2 fields contain data passed from the requesting thread to
  // the handler thread.

  // The thread ID of the thread requesting the dump (either the exception
  // thread or any other thread that called WriteMinidump directly).
  DWORD requesting_thread_id_;

  // The exception info passed to the exception handler on the exception
  // thread, if an exception occurred.  NULL for user-requested dumps.
  EXCEPTION_POINTERS* exception_info_;

  // If the handler is invoked due to an assertion, this will contain a
  // pointer to the assertion information.  It is NULL at other times.
  MDRawAssertionInfo* assertion_;

  // The return value of the handler, passed from the handler thread back to
  // the requesting thread.
  bool handler_return_value_;

  // If true, the handler will intercept EXCEPTION_BREAKPOINT and
  // EXCEPTION_SINGLE_STEP exceptions.  Leave this false (the default)
  // to not interfere with debuggers.
  bool handle_debug_exceptions_;

  // If true, the handler will consume any EXCEPTION_INVALID_HANDLE exceptions.
  // Leave this false (the default) to handle these exceptions as normal.
  bool consume_invalid_handle_exceptions_;

  // Callers can request additional memory regions to be included in
  // the dump.
  AppMemoryList app_memory_info_;

  // A stack of ExceptionHandler objects that have installed unhandled
  // exception filters.  This vector is used by HandleException to determine
  // which ExceptionHandler object to route an exception to.  When an
  // ExceptionHandler is created with install_handler true, it will append
  // itself to this list.
  static vector<ExceptionHandler*>* handler_stack_;

  // The index of the ExceptionHandler in handler_stack_ that will handle the
  // next exception.  Note that 0 means the last entry in handler_stack_, 1
  // means the next-to-last entry, and so on.  This is used by HandleException
  // to support multiple stacked Breakpad handlers.
  static LONG handler_stack_index_;

  // handler_stack_critical_section_ guards operations on handler_stack_ and
  // handler_stack_index_. The critical section is initialized by the
  // first instance of the class and destroyed by the last instance of it.
  static CRITICAL_SECTION handler_stack_critical_section_;

  // The number of instances of this class.
  static volatile LONG instance_count_;

  // disallow copy ctor and operator=
  explicit ExceptionHandler(const ExceptionHandler&);
  void operator=(const ExceptionHandler&);
};

}  // namespace google_breakpad

#pragma warning(pop)

#endif  // CLIENT_WINDOWS_HANDLER_EXCEPTION_HANDLER_H__
