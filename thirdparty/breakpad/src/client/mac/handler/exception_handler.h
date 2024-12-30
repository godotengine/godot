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

// exception_handler.h:  MacOS exception handler
// This class can install a Mach exception port handler to trap most common
// programming errors.  If an exception occurs, a minidump file will be
// generated which contains detailed information about the process and the
// exception.

#ifndef CLIENT_MAC_HANDLER_EXCEPTION_HANDLER_H__
#define CLIENT_MAC_HANDLER_EXCEPTION_HANDLER_H__

#include <mach/mach.h>
#include <TargetConditionals.h>

#include <string>

#include "client/mac/handler/ucontext_compat.h"
#include "common/scoped_ptr.h"

#if !TARGET_OS_IPHONE
#include "client/mac/crash_generation/crash_generation_client.h"
#endif

namespace google_breakpad {

using std::string;

struct ExceptionParameters;

enum HandlerThreadMessage {
  // Message ID telling the handler thread to write a dump.
  kWriteDumpMessage = 0,
  // Message ID telling the handler thread to write a dump and include
  // an exception stream.
  kWriteDumpWithExceptionMessage = 1,
  // Message ID telling the handler thread to quit.
  kShutdownMessage = 2
};

class ExceptionHandler {
 public:
  // A callback function to run before Breakpad performs any substantial
  // processing of an exception.  A FilterCallback is called before writing
  // a minidump.  context is the parameter supplied by the user as
  // callback_context when the handler was created.
  //
  // If a FilterCallback returns true, Breakpad will continue processing,
  // attempting to write a minidump.  If a FilterCallback returns false, Breakpad
  // will immediately report the exception as unhandled without writing a
  // minidump, allowing another handler the opportunity to handle it.
  typedef bool (*FilterCallback)(void* context);

  // A callback function to run after the minidump has been written.
  // |minidump_id| is a unique id for the dump, so the minidump
  // file is <dump_dir>/<minidump_id>.dmp.
  // |context| is the value passed into the constructor.
  // |succeeded| indicates whether a minidump file was successfully written.
  // Return true if the exception was fully handled and breakpad should exit.
  // Return false to allow any other exception handlers to process the
  // exception.
  typedef bool (*MinidumpCallback)(const char* dump_dir,
                                   const char* minidump_id,
                                   void* context, bool succeeded);

  // A callback function which will be called directly if an exception occurs.
  // This bypasses the minidump file writing and simply gives the client
  // the exception information.
  typedef bool (*DirectCallback)(void* context,
                                 int exception_type,
                                 int exception_code,
                                 int exception_subcode,
                                 mach_port_t thread_name);

  // Creates a new ExceptionHandler instance to handle writing minidumps.
  // Minidump files will be written to dump_path, and the optional callback
  // is called after writing the dump file, as described above.
  // If install_handler is true, then a minidump will be written whenever
  // an unhandled exception occurs.  If it is false, minidumps will only
  // be written when WriteMinidump is called.
  // If port_name is non-NULL, attempt to perform out-of-process dump generation
  // If port_name is NULL, in-process dump generation will be used.
  ExceptionHandler(const string& dump_path,
                   FilterCallback filter, MinidumpCallback callback,
                   void* callback_context, bool install_handler,
                   const char* port_name);

  // A special constructor if we want to bypass minidump writing and
  // simply get a callback with the exception information.
  ExceptionHandler(DirectCallback callback,
                   void* callback_context,
                   bool install_handler);

  ~ExceptionHandler();

  // Get and set the minidump path.
  string dump_path() const { return dump_path_; }
  void set_dump_path(const string& dump_path) {
    dump_path_ = dump_path;
    dump_path_c_ = dump_path_.c_str();
    UpdateNextID();  // Necessary to put dump_path_ in next_minidump_path_.
  }

  // Writes a minidump immediately.  This can be used to capture the
  // execution state independently of a crash.  Returns true on success.
  bool WriteMinidump() {
    return WriteMinidump(false);
  }

  bool WriteMinidump(bool write_exception_stream);

  // Convenience form of WriteMinidump which does not require an
  // ExceptionHandler instance.
  static bool WriteMinidump(const string& dump_path, MinidumpCallback callback,
                            void* callback_context) {
    return WriteMinidump(dump_path, false, callback, callback_context);
  }

  static bool WriteMinidump(const string& dump_path,
                            bool write_exception_stream,
                            MinidumpCallback callback,
                            void* callback_context);

  // Write a minidump of child immediately. This can be used to capture
  // the execution state of a child process independently of a crash.
  static bool WriteMinidumpForChild(mach_port_t child,
                                    mach_port_t child_blamed_thread,
                                    const std::string& dump_path,
                                    MinidumpCallback callback,
                                    void* callback_context);

  // Returns whether out-of-process dump generation is used or not.
  bool IsOutOfProcess() const {
#if TARGET_OS_IPHONE
    return false;
#else
    return crash_generation_client_.get() != NULL;
#endif
  }

 private:
  // Install the mach exception handler
  bool InstallHandler();

  // Uninstall the mach exception handler (if any)
  bool UninstallHandler(bool in_exception);

  // Setup the handler thread, and if |install_handler| is true, install the
  // mach exception port handler
  bool Setup(bool install_handler);

  // Uninstall the mach exception handler (if any) and terminate the helper
  // thread
  bool Teardown();

  // Send a mach message to the exception handler.  Return true on
  // success, false otherwise.
  bool SendMessageToHandlerThread(HandlerThreadMessage message_id);

  // All minidump writing goes through this one routine.
  // |task_context| can be NULL. If not, it will be used to retrieve the
  // context of the current thread, instead of using |thread_get_state|.
  bool WriteMinidumpWithException(int exception_type,
                                  int exception_code,
                                  int exception_subcode,
                                  breakpad_ucontext_t* task_context,
                                  mach_port_t thread_name,
                                  bool exit_after_write,
                                  bool report_current_thread);

  // When installed, this static function will be call from a newly created
  // pthread with |this| as the argument
  static void* WaitForMessage(void* exception_handler_class);

  // Signal handler for SIGABRT.
  static void SignalHandler(int sig, siginfo_t* info, void* uc);

  // disallow copy ctor and operator=
  explicit ExceptionHandler(const ExceptionHandler&);
  void operator=(const ExceptionHandler&);

  // Generates a new ID and stores it in next_minidump_id_, and stores the
  // path of the next minidump to be written in next_minidump_path_.
  void UpdateNextID();

  // These functions will suspend/resume all threads except for the
  // reporting thread
  bool SuspendThreads();
  bool ResumeThreads();

  // The destination directory for the minidump
  string dump_path_;

  // The basename of the next minidump w/o extension
  string next_minidump_id_;

  // The full path to the next minidump to be written, including extension
  string next_minidump_path_;

  // Pointers to the UTF-8 versions of above
  const char* dump_path_c_;
  const char* next_minidump_id_c_;
  const char* next_minidump_path_c_;

  // The callback function and pointer to be passed back after the minidump
  // has been written
  FilterCallback filter_;
  MinidumpCallback callback_;
  void* callback_context_;

  // The callback function to be passed back when we don't want a minidump
  // file to be written
  DirectCallback directCallback_;

  // The thread that is created for the handler
  pthread_t handler_thread_;

  // The port that is waiting on an exception message to be sent, if the
  // handler is installed
  mach_port_t handler_port_;

  // These variables save the previous exception handler's data so that it
  // can be re-installed when this handler is uninstalled
  ExceptionParameters* previous_;

  // True, if we've installed the exception handler
  bool installed_exception_handler_;

  // True, if we're in the process of uninstalling the exception handler and
  // the thread.
  bool is_in_teardown_;

  // Save the last result of the last minidump
  bool last_minidump_write_result_;

  // A mutex for use when writing out a minidump that was requested on a
  // thread other than the exception handler.
  pthread_mutex_t minidump_write_mutex_;

  // True, if we're using the mutext to indicate when mindump writing occurs
  bool use_minidump_write_mutex_;

  // Old signal handler for SIGABRT. Used to be able to restore it when
  // uninstalling.
  scoped_ptr<struct sigaction> old_handler_;

#if !TARGET_OS_IPHONE
  // Client for out-of-process dump generation.
  scoped_ptr<CrashGenerationClient> crash_generation_client_;
#endif
};

}  // namespace google_breakpad

#endif  // CLIENT_MAC_HANDLER_EXCEPTION_HANDLER_H__
