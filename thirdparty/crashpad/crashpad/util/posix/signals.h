// Copyright 2017 The Crashpad Authors. All rights reserved.
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

#ifndef CRASHPAD_UTIL_POSIX_SIGNALS_H_
#define CRASHPAD_UTIL_POSIX_SIGNALS_H_

#include <signal.h>

#include "base/macros.h"

namespace crashpad {

//! \brief Utilities for handling POSIX signals.
class Signals {
 public:
  //! \brief A signal number used by Crashpad to simulate signals.
  static constexpr int kSimulatedSigno = -1;

  //! \brief The type used for `struct sigaction::sa_sigaction`.
  using Handler = void (*)(int, siginfo_t*, void*);

  //! \brief A group of `struct sigaction` structures corresponding to a set
  //!     of signals’ previous actions, addressable by signal number.
  //!
  //! This type is used to store previous signal actions when new actions are
  //! installed in batch by InstallCrashHandlers() or
  //! InstallTerminateHandlers().
  //!
  //! This object is not initialized by any constructor. Its expected initial
  //! state is to have its contents filled with zeroes. Because signal handlers
  //! are stateless (there is no “context” parameter), any state must be
  //! accessed via objects of static storage duration, and it is expected that
  //! objects of this class will only ever exist with static storage duration,
  //! which in the absence of a constructor will be zero-initialized as
  //! expected. In the event that an object of this class must exist with a
  //! different storage duration, such as automatic or dynamic storage duration,
  //! it must be explicitly initialized. For example: `OldActions old_actions =
  //! {};`.
  class OldActions {
   public:
    // DISALLOW_COPY_AND_ASSIGN removes the default constructor, so explicitly
    // opt for it. This should not result in any static initialization code even
    // when an object of this class is given static storage duration.
    OldActions() = default;

    //! \brief Returns a `struct sigaction` structure corresponding to the
    //!     given signal.
    //!
    //! \note This method is safe to call from a signal handler.
    struct sigaction* ActionForSignal(int sig);

   private:
    // As a small storage optimization, don’t waste any space on a slot for
    // signal 0, because there is no signal 0.
    struct sigaction actions_[NSIG - 1];

    DISALLOW_COPY_AND_ASSIGN(OldActions);
  };

  //! \brief Installs a new signal handler.
  //!
  //! \param[in] sig The signal number to handle.
  //! \param[in] handler A signal-handling function to execute, used as the
  //!     `struct sigaction::sa_sigaction` field when calling `sigaction()`.
  //! \param[in] flags Flags to pass to `sigaction()` in the `struct
  //!     sigaction::sa_flags` field. `SA_SIGINFO` will be specified implicitly.
  //! \param[out] old_action The previous action for the signal, replaced by the
  //!     new action. May be `nullptr` if not needed.
  //!
  //! \return `true` on success. `false` on failure with a message logged.
  //!
  //! \warning This function may not be called from a signal handler because of
  //!     its use of logging. See RestoreHandlerAndReraiseSignalOnReturn()
  //!     instead.
  static bool InstallHandler(int sig,
                             Handler handler,
                             int flags,
                             struct sigaction* old_action);

  //! \brief Installs `SIG_DFL` for the signal \a sig.
  //!
  //! \param[in] sig The signal to set the default action for.
  //!
  //! \return `true` on success, `false` on failure with errno set. No message
  //!     is logged.
  static bool InstallDefaultHandler(int sig);

  //! \brief Installs a new signal handler for all signals associated with
  //!     crashes.
  //!
  //! Signals associated with crashes are those whose default dispositions
  //! involve creating a core dump. The precise set of signals involved varies
  //! between operating systems.
  //!
  //! A single signal may either be associated with a crash or with termination
  //! (see InstallTerminateHandlers()), and perhaps neither, but never both.
  //!
  //! \param[in] handler A signal-handling function to execute, used as the
  //!     `struct sigaction::sa_sigaction` field when calling `sigaction()`.
  //! \param[in] flags Flags to pass to `sigaction()` in the `struct
  //!     sigaction::sa_flags` field. `SA_SIGINFO` will be specified implicitly.
  //! \param[out] old_actions The previous actions for the signals, replaced by
  //!     the new action. May be `nullptr` if not needed. The same \a
  //!     old_actions object may be used for calls to both this function and
  //!     InstallTerminateHandlers().
  //!
  //! \return `true` on success. `false` on failure with a message logged.
  //!
  //! \warning This function may not be called from a signal handler because of
  //!     its use of logging. See RestoreHandlerAndReraiseSignalOnReturn()
  //!     instead.
  static bool InstallCrashHandlers(Handler handler,
                                   int flags,
                                   OldActions* old_actions);

  //! \brief Installs a new signal handler for all signals associated with
  //!     termination.
  //!
  //! Signals associated with termination are those whose default dispositions
  //! involve terminating the process without creating a core dump. The precise
  //! set of signals involved varies between operating systems.
  //!
  //! A single signal may either be associated with termination or with a
  //! crash (see InstalCrashHandlers()), and perhaps neither, but never both.
  //!
  //! \param[in] handler A signal-handling function to execute, used as the
  //!     `struct sigaction::sa_sigaction` field when calling `sigaction()`.
  //! \param[in] flags Flags to pass to `sigaction()` in the `struct
  //!     sigaction::sa_flags` field. `SA_SIGINFO` will be specified implicitly.
  //! \param[out] old_actions The previous actions for the signals, replaced by
  //!     the new action. May be `nullptr` if not needed. The same \a
  //!     old_actions object may be used for calls to both this function and
  //!     InstallCrashHandlers().
  //!
  //! \return `true` on success. `false` on failure with a message logged.
  //!
  //! \warning This function may not be called from a signal handler because of
  //!     its use of logging. See RestoreHandlerAndReraiseSignalOnReturn()
  //!     instead.
  static bool InstallTerminateHandlers(Handler handler,
                                       int flags,
                                       OldActions* old_actions);

  //! \brief Determines whether a signal will be re-raised autonomously upon
  //!     return from a signal handler.
  //!
  //! Certain signals, when generated synchronously in response to a hardware
  //! fault, are unrecoverable. Upon return from the signal handler, the same
  //! action that triggered the signal to be raised initially will be retried,
  //! and unless the signal handler took action to mitigate this error, the same
  //! signal will be re-raised. As an example, a CPU will not be able to read
  //! unmapped memory (causing `SIGSEGV`), thus the signal will be re-raised
  //! upon return from the signal handler unless the signal handler established
  //! a memory mapping where required.
  //!
  //! It is important to distinguish between these synchronous signals generated
  //! in response to a hardware fault and signals generated asynchronously or in
  //! software. As an example, `SIGSEGV` will not re-raise autonomously if sent
  //! by `kill()`.
  //!
  //! This function distinguishes between signals that can re-raise
  //! autonomously, and for those that can, between instances of the signal that
  //! were generated synchronously in response to a hardware fault and instances
  //! that were generated by other means.
  //!
  //! \param[in] siginfo A pointer to a `siginfo_t` object received by a signal
  //!     handler.
  //!
  //! \return `true` if the signal being handled will re-raise itself
  //!     autonomously upon return from a signal handler. `false` if it will
  //!     not. When this function returns `false`, a signal can still be
  //!     re-raised upon return from a signal handler by calling `raise()` from
  //!     within the signal handler.
  //!
  //! \note This function is safe to call from a signal handler.
  static bool WillSignalReraiseAutonomously(const siginfo_t* siginfo);

  //! \brief Restores a previous signal action and arranges to re-raise a signal
  //!     on return from a signal handler.
  //!
  //! \param[in] siginfo A pointer to a `siginfo_t` object received by a signal
  //!     handler.
  //! \param[in] old_action The previous action for the signal, which will be
  //!     re-established as the signal’s action. May be `nullptr`, which directs
  //!     the default action for the signal to be used.
  //!
  //! If this function fails, it will immediately call `_exit()` and set an exit
  //! status of `191`.
  //!
  //! \note This function may only be called from a signal handler.
  static void RestoreHandlerAndReraiseSignalOnReturn(
      const siginfo_t* siginfo,
      const struct sigaction* old_action);

  //! \brief Determines whether a signal is associated with a crash.
  //!
  //! Signals associated with crashes are those whose default dispositions
  //! involve creating a core dump. The precise set of signals involved varies
  //! between operating systems.
  //!
  //! \param[in] sig The signal to test.
  //!
  //! \return `true` if \a sig is associated with a crash. `false` otherwise.
  //!
  //! \note This function is safe to call from a signal handler.
  static bool IsCrashSignal(int sig);

  //! \brief Determines whether a signal is associated with termination.
  //!
  //! Signals associated with termination are those whose default dispositions
  //! involve terminating the process without creating a core dump. The precise
  //! set of signals involved varies between operating systems.
  //!
  //! \param[in] sig The signal to test.
  //!
  //! \return `true` if \a sig is associated with termination. `false`
  //!     otherwise.
  //!
  //! \note This function is safe to call from a signal handler.
  static bool IsTerminateSignal(int sig);

 private:
  DISALLOW_IMPLICIT_CONSTRUCTORS(Signals);
};

}  // namespace crashpad

#endif  // CRASHPAD_UTIL_POSIX_SIGNALS_H_
