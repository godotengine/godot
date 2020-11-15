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

#ifndef CRASHPAD_UTIL_WIN_SESSION_END_WATCHER_H_
#define CRASHPAD_UTIL_WIN_SESSION_END_WATCHER_H_

#include <windows.h>

#include "base/macros.h"
#include "util/thread/thread.h"
#include "util/win/scoped_handle.h"

namespace crashpad {

//! \brief Creates a hidden window and waits for a `WM_ENDSESSION` message,
//!     indicating that the session is ending and the application should
//!     terminate.
//!
//! A dedicated thread will be created to run the `GetMessage()`-based message
//! loop required to monitor for this message.
//!
//! Users should subclass this class and receive notifications by implementing
//! the SessionEndWatcherEvent() method.
class SessionEndWatcher : public Thread {
 public:
  SessionEndWatcher();

  //! \note The destructor waits for the thread that runs the message loop to
  //!     terminate.
  ~SessionEndWatcher() override;

 protected:
  // Exposed for testing.
  HWND GetWindow() const { return window_; }

  // Exposed for testing. Blocks until window_ has been created. May be called
  // multiple times if necessary.
  void WaitForStart();

  // Exposed for testing. Blocks until the message loop ends. May be called
  // multiple times if necessary.
  void WaitForStop();

 private:
  // Thread:
  void ThreadMain() override;

  static LRESULT CALLBACK WindowProc(HWND window,
                                     UINT message,
                                     WPARAM w_param,
                                     LPARAM l_param);

  //! \brief A `WM_ENDSESSION` message was received and it indicates that the
  //!     user session will be ending imminently.
  //!
  //! This method is called on the thread that runs the message loop.
  virtual void SessionEnding() = 0;

  HWND window_;  // Conceptually strong, but ownership managed in ThreadMain()
  ScopedKernelHANDLE started_;
  ScopedKernelHANDLE stopped_;

  DISALLOW_COPY_AND_ASSIGN(SessionEndWatcher);
};

}  // namespace crashpad

#endif  // CRASHPAD_UTIL_WIN_SESSION_END_WATCHER_H_
