//===- llvm/Support/Program.h ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the llvm::sys::Program class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_PROGRAM_H
#define LLVM_SUPPORT_PROGRAM_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/ErrorOr.h"
#include <system_error>

namespace llvm {
class StringRef;

namespace sys {

  /// This is the OS-specific separator for PATH like environment variables:
  // a colon on Unix or a semicolon on Windows.
#if defined(LLVM_ON_UNIX)
  const char EnvPathSeparator = ':';
#elif defined (LLVM_ON_WIN32)
  const char EnvPathSeparator = ';';
#endif

/// @brief This struct encapsulates information about a process.
struct ProcessInfo {
#if defined(LLVM_ON_UNIX)
  typedef pid_t ProcessId;
#elif defined(LLVM_ON_WIN32)
  typedef unsigned long ProcessId; // Must match the type of DWORD on Windows.
  typedef void * HANDLE; // Must match the type of HANDLE on Windows.
  /// The handle to the process (available on Windows only).
  HANDLE ProcessHandle;
#else
#error "ProcessInfo is not defined for this platform!"
#endif

  /// The process identifier.
  ProcessId Pid;

  /// The return code, set after execution.
  int ReturnCode;

  ProcessInfo();
};

  /// \brief Find the first executable file \p Name in \p Paths.
  ///
  /// This does not perform hashing as a shell would but instead stats each PATH
  /// entry individually so should generally be avoided. Core LLVM library
  /// functions and options should instead require fully specified paths.
  ///
  /// \param Name name of the executable to find. If it contains any system
  ///   slashes, it will be returned as is.
  /// \param Paths optional list of paths to search for \p Name. If empty it
  ///   will use the system PATH environment instead.
  ///
  /// \returns The fully qualified path to the first \p Name in \p Paths if it
  ///   exists. \p Name if \p Name has slashes in it. Otherwise an error.
  ErrorOr<std::string>
  findProgramByName(StringRef Name,
                    ArrayRef<StringRef> Paths = ArrayRef<StringRef>());

  // These functions change the specified standard stream (stdin or stdout) to
  // binary mode. They return errc::success if the specified stream
  // was changed. Otherwise a platform dependent error is returned.
  std::error_code ChangeStdinToBinary();
  std::error_code ChangeStdoutToBinary();

  /// This function executes the program using the arguments provided.  The
  /// invoked program will inherit the stdin, stdout, and stderr file
  /// descriptors, the environment and other configuration settings of the
  /// invoking program.
  /// This function waits for the program to finish, so should be avoided in
  /// library functions that aren't expected to block. Consider using
  /// ExecuteNoWait() instead.
  /// @returns an integer result code indicating the status of the program.
  /// A zero or positive value indicates the result code of the program.
  /// -1 indicates failure to execute
  /// -2 indicates a crash during execution or timeout
  int ExecuteAndWait(
      StringRef Program, ///< Path of the program to be executed. It is
      /// presumed this is the result of the findProgramByName method.
      const char **args, ///< A vector of strings that are passed to the
      ///< program.  The first element should be the name of the program.
      ///< The list *must* be terminated by a null char* entry.
      const char **env = nullptr, ///< An optional vector of strings to use for
      ///< the program's environment. If not provided, the current program's
      ///< environment will be used.
      const StringRef **redirects = nullptr, ///< An optional array of pointers
      ///< to paths. If the array is null, no redirection is done. The array
      ///< should have a size of at least three. The inferior process's
      ///< stdin(0), stdout(1), and stderr(2) will be redirected to the
      ///< corresponding paths.
      ///< When an empty path is passed in, the corresponding file
      ///< descriptor will be disconnected (ie, /dev/null'd) in a portable
      ///< way.
      unsigned secondsToWait = 0, ///< If non-zero, this specifies the amount
      ///< of time to wait for the child process to exit. If the time
      ///< expires, the child is killed and this call returns. If zero,
      ///< this function will wait until the child finishes or forever if
      ///< it doesn't.
      unsigned memoryLimit = 0, ///< If non-zero, this specifies max. amount
      ///< of memory can be allocated by process. If memory usage will be
      ///< higher limit, the child is killed and this call returns. If zero
      ///< - no memory limit.
      std::string *ErrMsg = nullptr, ///< If non-zero, provides a pointer to a
      ///< string instance in which error messages will be returned. If the
      ///< string is non-empty upon return an error occurred while invoking the
      ///< program.
      bool *ExecutionFailed = nullptr);

  /// Similar to ExecuteAndWait, but returns immediately.
  /// @returns The \see ProcessInfo of the newly launced process.
  /// \note On Microsoft Windows systems, users will need to either call \see
  /// Wait until the process finished execution or win32 CloseHandle() API on
  /// ProcessInfo.ProcessHandle to avoid memory leaks.
  ProcessInfo
  ExecuteNoWait(StringRef Program, const char **args, const char **env = nullptr,
                const StringRef **redirects = nullptr, unsigned memoryLimit = 0,
                std::string *ErrMsg = nullptr, bool *ExecutionFailed = nullptr);

  /// Return true if the given arguments fit within system-specific
  /// argument length limits.
  bool argumentsFitWithinSystemLimits(ArrayRef<const char*> Args);

  /// File encoding options when writing contents that a non-UTF8 tool will
  /// read (on Windows systems). For UNIX, we always use UTF-8.
  enum WindowsEncodingMethod {
    /// UTF-8 is the LLVM native encoding, being the same as "do not perform
    /// encoding conversion".
    WEM_UTF8,
    WEM_CurrentCodePage,
    WEM_UTF16
  };

  /// Saves the UTF8-encoded \p contents string into the file \p FileName
  /// using a specific encoding.
  ///
  /// This write file function adds the possibility to choose which encoding
  /// to use when writing a text file. On Windows, this is important when
  /// writing files with internationalization support with an encoding that is
  /// different from the one used in LLVM (UTF-8). We use this when writing
  /// response files, since GCC tools on MinGW only understand legacy code
  /// pages, and VisualStudio tools only understand UTF-16.
  /// For UNIX, using different encodings is silently ignored, since all tools
  /// work well with UTF-8.
  /// This function assumes that you only use UTF-8 *text* data and will convert
  /// it to your desired encoding before writing to the file.
  ///
  /// FIXME: We use EM_CurrentCodePage to write response files for GNU tools in
  /// a MinGW/MinGW-w64 environment, which has serious flaws but currently is
  /// our best shot to make gcc/ld understand international characters. This
  /// should be changed as soon as binutils fix this to support UTF16 on mingw.
  ///
  /// \returns non-zero error_code if failed
  std::error_code
  writeFileWithEncoding(StringRef FileName, StringRef Contents,
                        WindowsEncodingMethod Encoding = WEM_UTF8);

  /// This function waits for the process specified by \p PI to finish.
  /// \returns A \see ProcessInfo struct with Pid set to:
  /// \li The process id of the child process if the child process has changed
  /// state.
  /// \li 0 if the child process has not changed state.
  /// \note Users of this function should always check the ReturnCode member of
  /// the \see ProcessInfo returned from this function.
  ProcessInfo Wait(
      const ProcessInfo &PI, ///< The child process that should be waited on.
      unsigned SecondsToWait, ///< If non-zero, this specifies the amount of
      ///< time to wait for the child process to exit. If the time expires, the
      ///< child is killed and this function returns. If zero, this function
      ///< will perform a non-blocking wait on the child process.
      bool WaitUntilTerminates, ///< If true, ignores \p SecondsToWait and waits
      ///< until child has terminated.
      std::string *ErrMsg = nullptr ///< If non-zero, provides a pointer to a
      ///< string instance in which error messages will be returned. If the
      ///< string is non-empty upon return an error occurred while invoking the
      ///< program.
      );
  }
}

#endif
