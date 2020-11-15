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

#include "util/win/session_end_watcher.h"

#include "base/logging.h"
#include "base/scoped_generic.h"
#include "util/win/scoped_set_event.h"

extern "C" {
extern IMAGE_DOS_HEADER __ImageBase;
}  // extern "C"

namespace crashpad {

namespace {

// ScopedWindowClass and ScopedWindow operate on ATOM* and HWND*, respectively,
// instead of ATOM and HWND, so that the actual storage can exist as a local
// variable or a member variable, and the scoper can be responsible for
// releasing things only if the actual storage hasn’t been released and zeroed
// already by something else.
struct ScopedWindowClassTraits {
  static ATOM* InvalidValue() { return nullptr; }
  static void Free(ATOM* window_class) {
    if (*window_class) {
      if (!UnregisterClass(MAKEINTATOM(*window_class), 0)) {
        PLOG(ERROR) << "UnregisterClass";
      } else {
        *window_class = 0;
      }
    }
  }
};
using ScopedWindowClass = base::ScopedGeneric<ATOM*, ScopedWindowClassTraits>;

struct ScopedWindowTraits {
  static HWND* InvalidValue() { return nullptr; }
  static void Free(HWND* window) {
    if (*window) {
      if (!DestroyWindow(*window)) {
        PLOG(ERROR) << "DestroyWindow";
      } else {
        *window = nullptr;
      }
    }
  }
};
using ScopedWindow = base::ScopedGeneric<HWND*, ScopedWindowTraits>;

// GetWindowLongPtr()’s return value doesn’t unambiguously indicate whether it
// was successful, because 0 could either represent successful retrieval of the
// value 0, or failure. This wrapper is more convenient to use.
bool GetWindowLongPtrAndSuccess(HWND window, int index, LONG_PTR* value) {
  SetLastError(ERROR_SUCCESS);
  *value = GetWindowLongPtr(window, index);
  return *value || GetLastError() == ERROR_SUCCESS;
}

// SetWindowLongPtr() has the same problem as GetWindowLongPtr(). Use this
// wrapper instead.
bool SetWindowLongPtrAndGetSuccess(HWND window, int index, LONG_PTR value) {
  SetLastError(ERROR_SUCCESS);
  LONG_PTR previous = SetWindowLongPtr(window, index, value);
  return previous || GetLastError() == ERROR_SUCCESS;
}

}  // namespace

SessionEndWatcher::SessionEndWatcher()
    : Thread(),
      window_(nullptr),
      started_(nullptr),
      stopped_(nullptr) {
  // Set bManualReset for these events so that WaitForStart() and WaitForStop()
  // can be called multiple times.

  started_.reset(CreateEvent(nullptr, true, false, nullptr));
  PLOG_IF(ERROR, !started_.get()) << "CreateEvent";

  stopped_.reset(CreateEvent(nullptr, true, false, nullptr));
  PLOG_IF(ERROR, !stopped_.get()) << "CreateEvent";

  Start();
}

SessionEndWatcher::~SessionEndWatcher() {
  // Tear everything down by posting a WM_CLOSE to the window. This obviously
  // can’t work until the window has been created, and that happens on a
  // different thread, so wait for the start event to be signaled first.
  WaitForStart();
  if (window_) {
    if (!PostMessage(window_, WM_CLOSE, 0, 0)) {
      PLOG(ERROR) << "PostMessage";
    }
  }

  Join();
  DCHECK(!window_);
}

void SessionEndWatcher::WaitForStart() {
  if (WaitForSingleObject(started_.get(), INFINITE) != WAIT_OBJECT_0) {
    PLOG(ERROR) << "WaitForSingleObject";
  }
}

void SessionEndWatcher::WaitForStop() {
  if (WaitForSingleObject(stopped_.get(), INFINITE) != WAIT_OBJECT_0) {
    PLOG(ERROR) << "WaitForSingleObject";
  }
}

void SessionEndWatcher::ThreadMain() {
  ATOM atom = 0;
  ScopedWindowClass window_class(&atom);
  ScopedWindow window(&window_);

  ScopedSetEvent call_set_stop(stopped_.get());

  {
    ScopedSetEvent call_set_start(started_.get());

    WNDCLASS wndclass = {};
    wndclass.lpfnWndProc = WindowProc;
    wndclass.hInstance = reinterpret_cast<HMODULE>(&__ImageBase);
    wndclass.lpszClassName = L"crashpad_SessionEndWatcher";
    atom = RegisterClass(&wndclass);
    if (!atom) {
      PLOG(ERROR) << "RegisterClass";
      return;
    }

    window_ = CreateWindow(MAKEINTATOM(atom),  // lpClassName
                           nullptr,  // lpWindowName
                           0,  // dwStyle
                           0,  // x
                           0,  // y
                           0,  // nWidth
                           0,  // nHeight
                           nullptr,  // hWndParent
                           nullptr,  // hMenu
                           nullptr,  // hInstance
                           this);  // lpParam
    if (!window_) {
      PLOG(ERROR) << "CreateWindow";
      return;
    }
  }

  MSG message;
  BOOL rv = 0;
  while (window_ && (rv = GetMessage(&message, window_, 0, 0)) > 0) {
    TranslateMessage(&message);
    DispatchMessage(&message);
  }
  if (window_ && rv == -1) {
    PLOG(ERROR) << "GetMessage";
    return;
  }
}

// static
LRESULT CALLBACK SessionEndWatcher::WindowProc(HWND window,
                                               UINT message,
                                               WPARAM w_param,
                                               LPARAM l_param) {
  // Figure out which object this is. A pointer to it is stuffed into the last
  // parameter of CreateWindow(), which shows up as CREATESTRUCT::lpCreateParams
  // in a WM_CREATE message. That should be processed before any of the other
  // messages of interest to this function. Once the object is known, save a
  // pointer to it in the GWLP_USERDATA slot for later retrieval when processing
  // other messages.
  SessionEndWatcher* self;
  if (!GetWindowLongPtrAndSuccess(
          window, GWLP_USERDATA, reinterpret_cast<LONG_PTR*>(&self))) {
    PLOG(ERROR) << "GetWindowLongPtr";
  }
  if (!self && message == WM_CREATE) {
    CREATESTRUCT* create = reinterpret_cast<CREATESTRUCT*>(l_param);
    self = reinterpret_cast<SessionEndWatcher*>(create->lpCreateParams);
    if (!SetWindowLongPtrAndGetSuccess(
            window, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(self))) {
      PLOG(ERROR) << "SetWindowLongPtr";
    }
  }

  if (self) {
    if (message == WM_ENDSESSION) {
      // If w_param is false, this WM_ENDSESSION message cancels a previous
      // WM_QUERYENDSESSION.
      if (w_param) {
        self->SessionEnding();

        // If the session is ending, post a close message which will kick off
        // window destruction and cause the message loop thread to terminate.
        if (!PostMessage(self->window_, WM_CLOSE, 0, 0)) {
          PLOG(ERROR) << "PostMessage";
        }
      }
    } else if (message == WM_DESTROY) {
      // The window is being destroyed. Clear GWLP_USERDATA so that |self| won’t
      // be found during a subsequent call into this function for this window.
      // Clear self->window_ too, because it refers to an object that soon won’t
      // exist. That signals the message loop to stop processing messages.
      if (!SetWindowLongPtrAndGetSuccess(window, GWLP_USERDATA, 0)) {
        PLOG(ERROR) << "SetWindowLongPtr";
      }
      self->window_ = nullptr;
    }
  }

  // If the message is WM_CLOSE, DefWindowProc() will call DestroyWindow(), and
  // this function will be called again with a WM_DESTROY message.
  return DefWindowProc(window, message, w_param, l_param);
}

}  // namespace crashpad
