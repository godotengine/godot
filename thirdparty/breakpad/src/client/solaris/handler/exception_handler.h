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
//
// Author: Alfred Peng

#ifndef CLIENT_SOLARIS_HANDLER_EXCEPTION_HANDLER_H__
#define CLIENT_SOLARIS_HANDLER_EXCEPTION_HANDLER_H__

#include <map>
#include <string>
#include <vector>

#include "client/solaris/handler/minidump_generator.h"

namespace google_breakpad {

using std::string;

//
// ExceptionHandler
//
// ExceptionHandler can write a minidump file when an exception occurs,
// or when WriteMinidump() is called explicitly by your program.
//
// To have the exception handler write minidumps when an uncaught exception
// (crash) occurs, you should create an instance early in the execution
// of your program, and keep it around for the entire time you want to
// have crash handling active (typically, until shutdown).
// (NOTE): There should be only one this kind of exception handler
// object per process.
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
// Caller should try to make the callbacks as crash-friendly as possible,
// it should avoid use heap memory allocation as much as possible.
//
class ExceptionHandler {
 public:
  // A callback function to run before Breakpad performs any substantial
  // processing of an exception.  A FilterCallback is called before writing
  // a minidump.  context is the parameter supplied by the user as
  // callback_context when the handler was created.
  //
  // If a FilterCallback returns true, Breakpad will continue processing,
  // attempting to write a minidump.  If a FilterCallback returns false,
  // Breakpad  will immediately report the exception as unhandled without
  // writing a minidump, allowing another handler the opportunity to handle it.
  typedef bool (*FilterCallback)(void* context);

  // A callback function to run after the minidump has been written.
  // minidump_id is a unique id for the dump, so the minidump
  // file is <dump_path>/<minidump_id>.dmp.  context is the parameter supplied
  // by the user as callback_context when the handler was created.  succeeded
  // indicates whether a minidump file was successfully written.
  //
  // If an exception occurred and the callback returns true, Breakpad will
  // treat the exception as fully-handled, suppressing any other handlers from
  // being notified of the exception.  If the callback returns false, Breakpad
  // will treat the exception as unhandled, and allow another handler to handle
  // it. If there are no other handlers, Breakpad will report the exception to
  // the system as unhandled, allowing a debugger or native crash dialog the
  // opportunity to handle the exception.  Most callback implementations
  // should normally return the value of |succeeded|, or when they wish to
  // not report an exception of handled, false.  Callbacks will rarely want to
  // return true directly (unless |succeeded| is true).
  typedef bool (*MinidumpCallback)(const char* dump_path,
                                   const char* minidump_id,
                                   void* context,
                                   bool succeeded);

  // Creates a new ExceptionHandler instance to handle writing minidumps.
  // Before writing a minidump, the optional filter callback will be called.
  // Its return value determines whether or not Breakpad should write a
  // minidump.  Minidump files will be written to dump_path, and the optional
  // callback is called after writing the dump file, as described above.
  // If install_handler is true, then a minidump will be written whenever
  // an unhandled exception occurs.  If it is false, minidumps will only
  // be written when WriteMinidump is called.
  ExceptionHandler(const string& dump_path,
                   FilterCallback filter, MinidumpCallback callback,
                   void* callback_context,
                   bool install_handler);
  ~ExceptionHandler();

  // Get and Set the minidump path.
  string dump_path() const { return dump_path_; }
  void set_dump_path(const string& dump_path) {
    dump_path_ = dump_path;
    dump_path_c_ = dump_path_.c_str();
  }

  // Writes a minidump immediately.  This can be used to capture the
  // execution state independently of a crash.  Returns true on success.
  bool WriteMinidump();

  // Convenience form of WriteMinidump which does not require an
  // ExceptionHandler instance.
  static bool WriteMinidump(const string& dump_path,
                            MinidumpCallback callback,
                            void* callback_context);

 private:
  // Setup crash handler.
  void SetupHandler();
  // Setup signal handler for a signal.
  void SetupHandler(int signo);
  // Teardown the handler for a signal.
  void TeardownHandler(int signo);
  // Teardown all handlers.
  void TeardownAllHandlers();

  // Runs the main loop for the exception handler thread.
  static void* ExceptionHandlerThreadMain(void* lpParameter);

  // Signal handler.
  static void HandleException(int signo);

  // Write all the information to the dump file.
  // If called from a signal handler, sighandler_ebp is the ebp of
  // that signal handler's frame, and sig_ctx is an out parameter
  // that will be set to point at the ucontext_t that was placed
  // on the stack by the kernel.  You can pass zero and NULL
  // for the second and third parameters if you are not calling
  // this from a signal handler.
  bool InternalWriteMinidump(int signo, uintptr_t sighandler_ebp,
                             ucontext_t** sig_ctx);

 private:
  // The callbacks before and after writing the dump file.
  FilterCallback filter_;
  MinidumpCallback callback_;
  void* callback_context_;

  // The directory in which a minidump will be written, set by the dump_path
  // argument to the constructor, or set_dump_path.
  string dump_path_;
  // C style dump path. Keep this when setting dump path, since calling
  // c_str() of std::string when crashing may not be safe.
  const char* dump_path_c_;

  // True if the ExceptionHandler installed an unhandled exception filter
  // when created (with an install_handler parameter set to true).
  bool installed_handler_;

  // Keep the previous handlers for the signal.
  typedef void (*sighandler_t)(int);
  std::map<int, sighandler_t> old_handlers_;

  // The global exception handler stack. This is need becuase there may exist
  // multiple ExceptionHandler instances in a process. Each will have itself
  // registered in this stack.
  static std::vector<ExceptionHandler*>* handler_stack_;
  // The index of the handler that should handle the next exception.
  static int handler_stack_index_;
  static pthread_mutex_t handler_stack_mutex_;

  // The minidump generator.
  MinidumpGenerator minidump_generator_;

  // disallow copy ctor and operator=
  explicit ExceptionHandler(const ExceptionHandler&);
  void operator=(const ExceptionHandler&);
};

}  // namespace google_breakpad

#endif  // CLIENT_SOLARIS_HANDLER_EXCEPTION_HANDLER_H__
